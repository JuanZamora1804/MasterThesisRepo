from overlap_lightglue_utils import *
import pandas as pd
import numpy as np
from wakepy import keep

# Calculate overlap between two configurations of images
def overlap_lightglue_conf_vs_conf(conf1, conf2, save_plots=True, save_sorted=True):
    bridgesfm_img_dir = rf"C:\Users\juanj\Desktop\Bridges\{conf1}\Input images"
    # in_dir = Path(r"C:\Users\juanj\Desktop\bridgesfm\data\0_bridge_Gaskugel")
    in_dir = Path(rf"C:\Users\juanj\Desktop\Bridges\{conf1}")
    # in_dir = Path(r"C:\Users\juanj\Desktop\bridgesfm\data\0_bridge_Gaskugel\0. Previous version\2nd models - MOS")
    # img_list_file_1 = f"gaskugel_{conf1}.csv"
    # img_list_file_2 = f"gaskugel_{conf2}.csv"

    img_list_file_1 = f"{conf1}_image_list.csv"
    img_list_file_2 = img_list_file_1

    img_path_1 = in_dir / img_list_file_1
    img_path_2 = in_dir / img_list_file_2

    img_path_list_1 = []
    img_path_list_1 = pd.read_csv(img_path_1)["Filename"].tolist()
    n_images_1 = len(img_path_list_1)
    for i in range(n_images_1):
        img_path_list_1[i] = os.path.join(bridgesfm_img_dir, img_path_list_1[i])

    img_path_list_2 = []
    img_path_list_2 = pd.read_csv(img_path_2)["Filename"].tolist()
    n_images_2 = len(img_path_list_2)
    for i in range(n_images_2):
        img_path_list_2[i] = os.path.join(bridgesfm_img_dir, img_path_list_2[i])

    n_max = max(n_images_1, n_images_2)

    # if n_images_1 < n_max:
    #     print(
    #         f"{conf1} ({n_images_1} images) vs {conf2} ({n_images_2} images). Padding list."
    #     )
    #     img_path_list_1.extend(["-"] * (n_max - n_images_1))

    # if n_images_2 < n_max:
    #     print(
    #         f"{conf2} ({n_images_2} images) vs {conf1} ({n_images_1} images). Padding list."
    #     )
    #     img_path_list_2.extend(["-"] * (n_max - n_images_2))

    # Reorder lists: Common -> Unique -> Padding
    diff = set(img_path_list_1) ^ set(img_path_list_2)
    common = sorted(
        [img for img in set(img_path_list_1) & set(img_path_list_2) if img != "-"]
    )

    unique_1 = sorted([i for i in diff if i in img_path_list_1 and i != "-"])
    img_path_list_1 = common + unique_1 + ["-"] * (n_max - len(common) - len(unique_1))

    unique_2 = sorted([i for i in diff if i in img_path_list_2 and i != "-"])
    img_path_list_2 = common + unique_2 + ["-"] * (n_max - len(common) - len(unique_2))

    # Matrices for Convex Hull Method
    enclosure_matrix = np.zeros((n_max, n_max))
    concentration_matrix = np.zeros_like(enclosure_matrix)
    overlap_matrix = np.zeros_like(enclosure_matrix)
    evaluation_matrices_names = ["Enclosure", "Concentration", "Overlap"]
    evaluation_matrices = [enclosure_matrix, concentration_matrix, overlap_matrix]

    for i in range(n_max):
        
        if i % 10 == 0:
            print(f"Processing row {i} of {n_max}")

        image_path_x = img_path_list_1[i]
        if image_path_x == "-":
            continue

        image_x = load_image(image_path_x)
        for j in range(n_max):
            image_path_y = img_path_list_2[j]
            if image_path_y == "-":
                continue

            image_y = load_image(image_path_y)

            image_x_total_area = image_x.shape[1] * image_x.shape[2]
            image_y_total_area = image_y.shape[1] * image_y.shape[2]

            m_kpts0, m_kpts1, _ = extract_matched_keypoints(image_x, image_y)

            # Convex Hull Method
            ratioA, ratioB = overlap_convex_hull(
                m_kpts0, m_kpts1, image_x_total_area, image_y_total_area
            )

            enclosure_matrix[i, j] = ratioA
            concentration_matrix[i, j] = ratioB
            overlap_matrix[i, j] = np.mean([ratioA, ratioB])
          
    # for matrix_name in evaluation_matrices_names:
    #     matrix = evaluation_matrices[evaluation_matrices_names.index(matrix_name)]
    #     csv_file_name = f"{matrix_name}_{conf1}.csv"
    #     save_matrix_as_csv(matrix, img_path_list_1, img_path_list_2, csv_file_name)
    #     print(f"Saved {csv_file_name}")

    for matrix_name in evaluation_matrices_names:
        matrix = evaluation_matrices[evaluation_matrices_names.index(matrix_name)]
        csv_file_name = f"{conf1}_{matrix_name}.csv"
        n_characters = len(rf"C:\Users\juanj\Desktop\Bridges\{conf1}\Input images") + 1
        save_matrix_as_csv(n_characters, matrix, img_path_list_1, img_path_list_2, csv_file_name, save_matrix_path = in_dir)
        print(f"Saved {csv_file_name}")

    print_metrics_confs(
        conf1, conf2, enclosure_matrix, concentration_matrix, overlap_matrix, rf"C:\Users\juanj\Desktop\Bridges\{conf1}\{conf1}_Overlap_Metrics.csv"
    )

    if save_plots:
        for matrix_name in evaluation_matrices_names:
            matrix = evaluation_matrices[evaluation_matrices_names.index(matrix_name)]
            heatmap_and_histogram_matrix(
                img_path_list_1, img_path_list_2, matrix, matrix_name, conf1, conf2
            )
            print(f"Plotted {matrix_name} heatmap and histogram.")

    # # Sort based on average overlap
    # row_means = np.mean(overlap_matrix, axis=1)
    # col_means = np.mean(overlap_matrix, axis=0)

    # # Sort indices descending
    # idx_rows = np.argsort(row_means)[::-1]
    # idx_cols = np.argsort(col_means)[::-1]

    # # Reorder image lists
    # sorted_img_list_1 = [img_path_list_1[i] for i in idx_rows]
    # sorted_img_list_2 = [img_path_list_2[i] for i in idx_cols]

    # os.makedirs("./sorting_results", exist_ok=True)  # Ensure directory exists
    # sorted_lists_file = f"./sorting_results/sorted_images_{conf1}_vs_{conf2}.csv"
    # df_sorted = pd.DataFrame(
    #     {f"{conf1}_Sorted": sorted_img_list_1, f"{conf2}_Sorted": sorted_img_list_2}
    # )
    # df_sorted.to_csv(sorted_lists_file, index=False)

    # if save_sorted:
    #     for matrix_name in evaluation_matrices_names:
    #         matrix = evaluation_matrices[evaluation_matrices_names.index(matrix_name)]
    #         sorted_matrix = matrix[idx_rows, :][:, idx_cols]

    #         heatmap_and_histogram_matrix(
    #             sorted_img_list_1,
    #             sorted_img_list_2,
    #             sorted_matrix,
    #             f"{matrix_name} Sorted",
    #             conf1,
    #             conf2,
    #         )


# Calculate overlap between one image and a configuration of images
def overlap_lightglue_img_vs_conf(conf1, conf2, save_plots=True):
    """
    config1 and config2 are strings representing different configurations
    e.g., 'Conf_10', 'Conf_11', etc.

    Also,
    conf1: Baseline configuration (e.g., '10')
    conf2: New configuration containing the added image (e.g., '11')
    """
    bridgesfm_img_dir = "..\\bridgesfm"
    in_dir = Path(r"C:\Users\juanj\Desktop\bridgesfm\data\0_bridge_Gaskugel")
    img_list_file_1 = f"gaskugel_{conf1}.csv"
    img_list_file_2 = f"gaskugel_{conf2}.csv"

    img_path_1 = in_dir / img_list_file_1
    img_path_2 = in_dir / img_list_file_2

    img_path_list_1 = []
    with open(img_path_1, "r") as f:
        for line in f:
            img_path_list_1.append(line.strip())

    img_path_list_2 = []
    with open(img_path_2, "r") as f:
        for line in f:
            img_path_list_2.append(line.strip())

    # Find the added image(s) using symmetric difference
    added_images_names = list(set(img_path_list_2) ^ set(img_path_list_1))

    if not added_images_names:
        print("No difference found between configurations.")
        return
    else:
        print(f"Added image(s) found: {added_images_names}")
        n_images_1 = len(img_path_list_1)
        for i in range(n_images_1):
            img_path_list_1[i] = os.path.join(bridgesfm_img_dir, img_path_list_1[i])

        added_images_full = [
            os.path.join(bridgesfm_img_dir, p) for p in added_images_names
        ]
        n_added = len(added_images_full)

        print(f"Comparing {n_added} new image(s) against {n_images_1} baseline images.")

        enclosure_matrix = np.zeros((n_added, n_images_1))
        concentration_matrix = np.zeros_like(enclosure_matrix)
        overlap_matrix = np.zeros_like(enclosure_matrix)

        evaluation_matrices_names = ["Enclosure", "Concentration", "Overlap"]
        evaluation_matrices = [enclosure_matrix, concentration_matrix, overlap_matrix]

        for i in range(n_added):
            image_path_x = added_images_full[i]
            image_x = load_image(image_path_x)

            for j in range(n_images_1):
                image_path_y = img_path_list_1[j]
                image_y = load_image(image_path_y)

                image_x_total_area = image_x.shape[1] * image_x.shape[2]
                image_y_total_area = image_y.shape[1] * image_y.shape[2]

                m_kpts0, m_kpts1, _ = extract_matched_keypoints(image_x, image_y)

                # Convex Hull Method
                ratioA, ratioB = overlap_convex_hull(
                    m_kpts0, m_kpts1, image_x_total_area, image_y_total_area
                )

                enclosure_matrix[i, j] = ratioA
                concentration_matrix[i, j] = ratioB

        for i in range(n_added):
            for j in range(n_images_1):
                overlap_matrix[i, j] = np.mean(
                    [enclosure_matrix[i, j], concentration_matrix[i, j]]
                )

        print_metrics_added_img(
            added_images_names[0][33:],
            conf2,
            enclosure_matrix,
            concentration_matrix,
            overlap_matrix,
        )  # 33: is for gaskugel_

        if save_plots:
            for matrix_name in evaluation_matrices_names:
                matrix = evaluation_matrices[
                    evaluation_matrices_names.index(matrix_name)
                ]
                heatmap_and_histogram_vector(
                    added_images_full,
                    img_path_list_1,
                    matrix,
                    matrix_name,
                    added_images_names[0][33:],
                    conf2,
                )


# Compare configurations vs configurations
# for i in range(0, 2):
#     basic_conf = "Conf_0"
#     conf = f"Conf_{i}"
#     print(f"Processing {basic_conf} vs {conf}")
#     overlap_lightglue_conf_vs_conf(
#         basic_conf, conf, save_plots=True, save_sorted=True
#     )

# Compare images vs configurations. Conf_i should have all the images on Conf_0 and some more.
# for i in range(48):
#     os.system("cls")
#     print(f"Processing Conf_0 vs Conf_{i}")
#     overlap_lightglue_img_vs_conf("Conf_0", f"Conf_{i}", save_plots=False)

def main():
        
    with keep.running():
        folders = []
        for entry in os.listdir(r"C:\Users\juanj\Desktop\Bridges"):
            if os.path.isdir(os.path.join(r"C:\Users\juanj\Desktop\Bridges", entry)):
                folders.append(entry)

        for folder in folders:
            basic_conf = folder
            conf = basic_conf
            print(f"Processing {basic_conf}")
            overlap_lightglue_conf_vs_conf(
                basic_conf, conf, save_plots=False, save_sorted=False
        )
            
if __name__ == "__main__":
    main()