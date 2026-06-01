import pandas as pd
import numpy as np
from overlap_lightglue_utils import *
from overlap_lightglue_analysis import overlap_lightglue_conf_vs_conf
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from pathlib import Path
import shutil
import os
from PIL import Image
from wakepy import keep


# 1. Make first set of images based on OpenCLIP's classification
def filtered_subset(
    df,
    scale_filter,
    scale_prob_filter,
    prefix,
    save_file=False,
    filename="filtered_subset.csv",
):
    if isinstance(scale_filter, list):
        try:
            df = df[df["scale"].isin(scale_filter)]
        except KeyError:
            print(f"'scale' column not found in DataFrame.")
            return pd.DataFrame()

    if scale_prob_filter > 0:
        try:
            df = df[df["scale_probabilities"] >= scale_prob_filter]
        except KeyError:
            print(f"'scale_probabilities' column not found in DataFrame.")
            return pd.DataFrame()

    df = df.copy()  # avoid SettingWithCopyWarning
    df["Filename"] = df["Filename"].apply(lambda f: os.path.join(prefix, str(f)))

    if save_file:
        df["Filename"].to_csv(filename, index=False, header="Filename")
        print(f"Image list file changed to the reduced subset of images.")

    print(f"{len(df)} images filtered.\n")
    print("=" * 20)
    print("Scale distribution after filtering:")
    print(df["scale"].value_counts())
    print("=" * 20)
    print("Position distribution after filtering:")
    print(df["position"].value_counts())
    print("=" * 20)

    return df


def return_metrics(df):
    print(df["scale"].value_counts())
    print("=" * 20)
    print(df["position"].value_counts())
    print("=" * 20)


# 2. Delete images that do not belong to the data set. Ex: images bellow a certain TH


def clean_img_list(images_list, prefix):
    for i, image in enumerate(images_list):
        images_list[i] = image.replace(prefix, "").lstrip("\\/")

    return images_list


def filter_minimum_th(full_overlap_matrix, images_list, minimum_th):
    print("Applying minimum overlap score threshold to filter images")

    overlap_matrix = full_overlap_matrix.loc[images_list, images_list]
    diagonal_mask = np.eye(overlap_matrix.shape[0], dtype=bool)
    overlap_matrix[diagonal_mask] = 0
    max_overlaps = overlap_matrix.max(axis=1)

    while (max_overlaps <= minimum_th).any():
        to_remove = max_overlaps[max_overlaps <= minimum_th].index.tolist()
        reduced_img_list = [img for img in overlap_matrix.index if img not in to_remove]
        overlap_matrix = overlap_matrix.loc[
            reduced_img_list, reduced_img_list
        ]  # ← add this
        max_overlaps = overlap_matrix.max(axis=1)

    print(
        f"Initial {len(images_list)} images reduced to {len(max_overlaps)} images with a minimum overlap threshold of {minimum_th:.2f}%.\n"
    )

    min_th_filtered = max_overlaps.index.tolist()

    return min_th_filtered


# 3. Use TSP to delete similar/duplicates.
def create_data_model(overlap_results):
    data = {}

    # Accept either a CSV path or a DataFrame-like object
    if isinstance(overlap_results, (str, Path)):
        df = pd.read_csv(overlap_results, index_col=0)
    else:
        df = pd.DataFrame(overlap_results)

    # If square with matching index/columns, use full matrix
    if list(df.columns) == list(df.index):
        labels_images = df.columns.tolist()
        overlap_matrix = df.values
    else:
        # Handle CSVs where the first column duplicated the index (no index_col used)
        first_col = df.iloc[:, 0]
        if first_col.equals(pd.Series(df.index, index=df.index)):
            labels_images = df.columns[1:].tolist()
            overlap_matrix = df.iloc[:, 1:].values
        else:
            labels_images = df.columns.tolist()
            overlap_matrix = df.values

    distance_matrix = (np.ones(overlap_matrix.shape) * 100 - overlap_matrix).astype(int)
    np.fill_diagonal(distance_matrix, 0)

    positive_distances = distance_matrix[distance_matrix > 0]
    if positive_distances.size > 0:
        min_dist = positive_distances.min()
        min_row = int(np.where(distance_matrix == min_dist)[0][0])
    else:
        min_row = 0

    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    data["depot"] = min_row

    return data, labels_images


def get_solution_indices(manager, routing, solution, distance_matrix):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()} of overlap score")
    index = routing.Start(0)
    indices = []
    route_info = []  # Store (from, to, overlap) tuples

    plan_output = (
        "Chosen images sequence (position according to the predefined configuration):\n"
    )
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        from_node = manager.IndexToNode(index)
        indices.append(from_node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        to_node = manager.IndexToNode(index)

        # Calculate overlap score from distance
        distance = routing.GetArcCostForVehicle(previous_index, index, 0)
        overlap_score = 100 - distance_matrix[from_node][to_node]

        route_info.append((from_node, to_node, overlap_score))
        route_distance += distance

    plan_output += f" {manager.IndexToNode(index)}\n"
    # print(plan_output)
    plan_output += f"Overlap 'distance': {route_distance} of overlap score"

    total_overlap_score = (len(indices) * 100) - route_distance
    # print(
    #     f"Total Chain Overlap Score: {total_overlap_score}\nMax Possible (Idealistic) Score: {len(indices) * 100}\n"
    # )

    return indices, route_info, total_overlap_score


def apply_tsp(overlap_matrix, subprocess, save_path):

    data, labels_images = create_data_model(overlap_matrix)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        indices, route_info, total_overlap_score = get_solution_indices(
            manager, routing, solution, data["distance_matrix"]
        )

        plot_matrix_solution(
            manager,
            routing,
            solution,
            100 - data["distance_matrix"],
            rf"{save_path}\{subprocess}_TSP_solution_matrix.png",
        )

        return indices, route_info, total_overlap_score, labels_images, data

    else:
        raise RuntimeError(
            f"TSP solver found no solution for the given overlap matrix."
        )


def filter_maximum_th(full_overlap_matrix, images_list, maximum_th, save_path):

    overlap_matrix = full_overlap_matrix.loc[images_list, images_list]

    print("Applying TSP to filter duplicates or similar images")

    indices, route_info, total_overlap_score, labels_images, data = apply_tsp(
        overlap_matrix, subprocess="MaxTH", save_path=save_path
    )

    # Create DataFrame with all columns
    filenames = [labels_images[idx] for idx in indices]
    overlap_scores = [info[2] for info in route_info]

    first_tsp_solution = pd.DataFrame(
        {
            "Filename": filenames,
            "Overlap_score": overlap_scores,
        }
    )

    for i in range(len(first_tsp_solution) - 1):
        if (
            first_tsp_solution.loc[i, "Overlap_score"] >= maximum_th
        ):  # Threshold for considering images as duplicates
            first_tsp_solution.loc[i + 1, "Filename"] = np.nan  # Mark as duplicate

    second_tsp_solution = first_tsp_solution.dropna(subset=["Filename"]).reset_index(
        drop=True
    )

    second_tsp_solution = second_tsp_solution["Filename"].tolist()

    # Run TSP again on the reduced set to ensure no remaining duplicates
    overlap_matrix_reduced = full_overlap_matrix.loc[
        second_tsp_solution, second_tsp_solution
    ]

    (
        indices_reduced,
        route_info_reduced,
        total_overlap_score_reduced,
        labels_images_reduced,
        data_reduced,
    ) = apply_tsp(overlap_matrix_reduced, subprocess="MaxTH", save_path=save_path)

    filenames_reduced = [labels_images_reduced[idx] for idx in indices_reduced]

    if len(filenames_reduced) == len(second_tsp_solution):
        print("Reduction successful with no remaining duplicates.")
    else:
        print("Duplicates may still be present after second TSP application.")

    print(
        f"Reduced {len(first_tsp_solution)} images reduced again to {len(second_tsp_solution)} images. With a maximum overlap threshold of {maximum_th:.2f}%.\n"
    )

    return second_tsp_solution


def save_files(file_list, dest_dir, images_dir):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for image in file_list:
        src = Path(images_dir) / image
        if not src.exists():
            print(f"File not found: {src}")
            continue
        try:
            shutil.copy2(src, dest / src.name)
            # print(f"Copied: {src} to {dest / src.name}")
            # print(f"Progress: {file_list.index(image) + 1}/{len(file_list)}")
        except Exception as e:
            print(f"Failed to copy {src}: {e}")



# #######################################

# ## Apply
# with keep.running():
#     folders = []
#     for entry in os.listdir(r"C:\Users\juanj\Desktop\Bridges"):
#         if os.path.isdir(os.path.join(r"C:\Users\juanj\Desktop\Bridges", entry)):
#             folders.append(entry)

#     for folder in folders:
#         bridge_name = folder
#         print(f"Processing bridge: {bridge_name}")
#         subsetfile_name = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_image_list.csv"
#         prefix = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images"

#         # Read classification of images with OpenCLIP
#         openclip_df = pd.read_csv(
#             rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_classification_results_with_CLIP_ViT-B-16_laion2b_s34b_b88k.csv"  # Model for scale only.
#         ).drop(columns=["index"])
#         openclip_df = openclip_df.rename(columns={"filename": "Filename"})

#         # Filter based on a scale and its probability
#         scale_filter = ["overall", "macro"]
#         scale_prob_filter = 0.4

#         subset = filtered_subset(
#             openclip_df,
#             scale_filter,
#             scale_prob_filter,
#             prefix,
#             save_file=True,
#             filename=subsetfile_name,
#         )
#         subset = clean_img_list(subset["Filename"].tolist(), prefix)
#         # subset.to_csv(subsetfile_name, index=False, header=False)

#         # Create Overlap matrix with the reduced set of images after OpenCLIP's classification
#         basic_conf = folder
#         conf = basic_conf
#         print(f"Processing {basic_conf}")
#         overlap_lightglue_conf_vs_conf(
#             basic_conf, conf, save_plots=False, save_sorted=False
#         )

#         full_overlap_matrix = pd.read_csv(
#             rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_Overlap.csv",
#             index_col=0,
#         )

#         min_th = 20
#         max_th = 50

#         # Filter based on minimum overlap score
#         min_filtered_list = filter_minimum_th(full_overlap_matrix, subset, min_th)
#         # print(min_filtered_list)

#         max_filtered_list = filter_maximum_th(
#             full_overlap_matrix,
#             min_filtered_list,
#             max_th,
#             save_path=rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}",
#         )
#         # print(max_filtered_list)

#         save_files(
#             max_filtered_list,
#             rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_filtered_images_{'_'.join(scale_filter)}",
#             rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images",
#         )

#         max_filtered_list = pd.DataFrame({"Filename": max_filtered_list})
#         max_filtered_list["Filename"] = (
#             f"C:\\Users\\juanj\\Desktop\\Bridges\\{bridge_name}\\{bridge_name}_filtered_images_{'_'.join(scale_filter)}"
#             + "\\"
#             + max_filtered_list["Filename"].astype(str)
#         )
#         max_filtered_list.to_csv(
#             rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_filtered_images_{'_'.join(scale_filter)}.csv",
#             index=False,
#             header=False,
#         )


# #######################################

with keep.running():
    folders = []
    for entry in os.listdir(r"C:\Users\juanj\Desktop\Bridges"):
        if os.path.isdir(os.path.join(r"C:\Users\juanj\Desktop\Bridges", entry)):
            folders.append(entry)

    for folder in folders:
        bridge_name = folder
        print(f"Processing bridge: {bridge_name}")
        subsetfile_name = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_image_list.csv"
        prefix = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images"

        openclip_df = pd.read_csv(
            rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_classification_results_with_CLIP_ViT-B-16_laion2b_s34b_b88k.csv"
        ).drop(columns=["index"])
        openclip_df = openclip_df.rename(columns={"filename": "Filename"})

        scale_filter = ["overall", "macro", "meso", "micro"] #["overall", "macro"] # 
        scale_prob_filter = 0.01

        subset = filtered_subset(
            openclip_df,
            scale_filter,
            scale_prob_filter,
            prefix,
            save_file=True,
            filename=subsetfile_name,
        )
        subset = clean_img_list(subset["Filename"].tolist(), prefix)

        basic_conf = folder
        conf = basic_conf
        print(f"Processing {basic_conf}")
        overlap_lightglue_conf_vs_conf(
            basic_conf, conf, save_plots=False, save_sorted=False
        )

        full_overlap_matrix = pd.read_csv(
            rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_Overlap.csv",
            index_col=0,
        )

        # --- Interactive threshold loop ---
        threshold_log_path = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_threshold_log.csv"

        # Initialize log file with header if it doesn't exist
        if not os.path.exists(threshold_log_path):
            pd.DataFrame(columns=["bridge", "min_th", "max_th", "n_after_min", "n_after_max"]).to_csv(
                threshold_log_path, index=False
            )

        while True:
            raw_min = input(f"\n[{bridge_name}] Enter min_th (default 20): ").strip()
            raw_max = input(f"[{bridge_name}] Enter max_th (default 50): ").strip()

            min_th = float(raw_min) if raw_min else 20.0
            max_th = float(raw_max) if raw_max else 50.0

            min_filtered_list = filter_minimum_th(full_overlap_matrix, subset, min_th)
            print(f"  → {len(min_filtered_list)} images after min_th={min_th}")

            max_filtered_list = filter_maximum_th(
                full_overlap_matrix,
                min_filtered_list,
                max_th,
                save_path=rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}",
            )
            print(f"  → {len(max_filtered_list)} images after max_th={max_th}")

            confirm = input("Accept this result? [y/n]: ").strip().lower()

            # Append this attempt to the log regardless of acceptance
            log_entry = pd.DataFrame([{
                "bridge": bridge_name,
                "min_th": min_th,
                "max_th": max_th,
                "n_after_min": len(min_filtered_list),
                "n_after_max": len(max_filtered_list),
            }])
            log_entry.to_csv(threshold_log_path, mode="a", header=False, index=False)

            if confirm == "y":
                print(f"Thresholds accepted: min_th={min_th}, max_th={max_th}")
                break
            else:
                print("Retrying with new thresholds...\n")
        # --- End of loop ---

        save_files(
            max_filtered_list,
            rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_filtered_images_{'_'.join(scale_filter)}",
            rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images",
        )

        max_filtered_list = pd.DataFrame({"Filename": max_filtered_list})
        max_filtered_list["Filename"] = (
            f"C:\\Users\\juanj\\Desktop\\Bridges\\{bridge_name}\\{bridge_name}_filtered_images_{'_'.join(scale_filter)}"
            + "\\"
            + max_filtered_list["Filename"].astype(str)
        )
        max_filtered_list.to_csv(
            rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_filtered_images_{'_'.join(scale_filter)}.csv",
            index=False,
            header=False,
        )