from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
plt.rcParams['font.family'] = 'Times New Roman'


def extract_matched_keypoints(
    image0, image1, extractor=None, matcher=None, device=None
):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=500000).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Keypoints and matches
    kpts0, kpts1, matches = (
        feats0["keypoints"],
        feats1["keypoints"],
        matches01["matches"],
    )

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return m_kpts0, m_kpts1, matches


def convex_hull_area(matched_keypoints):
    matched_keypoints_np = matched_keypoints.cpu().numpy()
    if matched_keypoints_np.ndim != 2 or matched_keypoints_np.shape[0] < 3:
        return 0.0

    try:
        hull = ConvexHull(matched_keypoints_np)
        return hull.volume
    except Exception:
        return 0.0


def overlap_convex_hull(m_kpts0, m_kpts1, image0_total_area, image1_total_area):
    ratio0 = convex_hull_area(m_kpts0) / image0_total_area * 100
    ratio1 = convex_hull_area(m_kpts1) / image1_total_area * 100

    return ratio0, ratio1


def matrix_without_diagonal(matrix):
    off_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[off_diagonal_mask]


def calculate_metrics(matrix):
    filtered_data = matrix[matrix < 90.0]

    if filtered_data.size == 0:
        return 0, 0, 0, 0, 0, 0

    _min = np.min(filtered_data).round(3)
    _max = np.max(filtered_data).round(3)
    median = np.median(filtered_data).round(3)
    mean = np.mean(filtered_data, dtype=np.float64).round(3)
    std = np.std(filtered_data).round(3)
    var = np.var(filtered_data).round(3)

    return _min, _max, median, mean, std, var

def save_matrix_as_csv(n_characters, matrix, x_labels, y_labels, file_name, save_matrix_path = "Results overlap calculations"):
    # Gaskugel
    # x_labels = [label[46:] for label in x_labels]
    # y_labels = [label[46:] for label in y_labels]

    # Nibelungen
    # x_labels = [label[48:] for label in x_labels]
    # y_labels = [label[48:] for label in y_labels]

    x_labels = [label[n_characters:] for label in x_labels]
    y_labels = [label[n_characters:] for label in y_labels]
    matrix = np.round(matrix, 5)
    df = pd.DataFrame(matrix, index=x_labels, columns=y_labels)
    df.to_csv(Path(save_matrix_path) / file_name)


def print_metrics_confs(config1, config2, enclosure, concentration, overlap, save_file_path):

    enclosure = matrix_without_diagonal(enclosure)
    concentration = matrix_without_diagonal(concentration)
    overlap = matrix_without_diagonal(overlap)

    (
        min_enclosu,
        max_enclosu,
        median_enclosu,
        mean_enclosu,
        std_enclosu,
        var_enclosu,
    ) = calculate_metrics(enclosure)
    (
        min_concent,
        max_concent,
        median_concent,
        mean_concent,
        std_concent,
        var_concent,
    ) = calculate_metrics(concentration)
    (
        min_overlap,
        max_overlap,
        median_overlap,
        mean_overlap,
        std_overlap,
        var_overlap,
    ) = calculate_metrics(overlap)

    file_name = save_file_path
    file_exists = os.path.isfile(file_name)

    results = pd.DataFrame(
        {
            "Config X": [config1],
            "Config Y": [config2],
            "Encl_max": [round(max_enclosu, 2)],
            "Encl_min": [round(min_enclosu, 2)],
            "Encl_median": [round(median_enclosu, 2)],
            "Encl_mean": [round(mean_enclosu, 2)],
            "Encl_std": [round(std_enclosu, 2)],
            "Encl_var": [round(var_enclosu, 2)],
            "Conc_max": [round(max_concent, 2)],
            "Conc_min": [round(min_concent, 2)],
            "Conc_median": [round(median_concent, 2)],
            "Conc_mean": [round(mean_concent, 2)],
            "Conc_std": [round(std_concent, 2)],
            "Conc_var": [round(var_concent, 2)],
            "Overlap_max": [round(max_overlap, 2)],
            "Overlap_min": [round(min_overlap, 2)],
            "Overlap_median": [round(median_overlap, 2)],
            "Overlap_mean": [round(mean_overlap, 2)],
            "Overlap_std": [round(std_overlap, 2)],
            "Overlap_var": [round(var_overlap, 2)],
        }
    )

    results.to_csv(file_name, mode="a", header=not file_exists, index=False)


def print_metrics_added_img(
    added_imags_names, config2, enclosure, concentration, overlap
):

    min_enclosu, max_enclosu, median_enclosu, mean_enclosu, std_enclosu, var_enclosu = (
        calculate_metrics(enclosure)
    )
    min_concent, max_concent, median_concent, mean_concent, std_concent, var_concent = (
        calculate_metrics(concentration)
    )
    min_overlap, max_overlap, median_overlap, mean_overlap, std_overlap, var_overlap = (
        calculate_metrics(overlap)
    )

    file_name = rf"../Image_vs_Config_Overlap_Metrics.csv"
    file_exists = os.path.isfile(file_name)

    results = pd.DataFrame(
        {
            "Added images": [added_imags_names],
            "Config": [config2],
            "Encl_max": [round(max_enclosu, 2)],
            "Encl_min": [round(min_enclosu, 2)],
            "Encl_median": [round(median_enclosu, 2)],
            "Encl_mean": [round(mean_enclosu, 2)],
            "Encl_std": [round(std_enclosu, 2)],
            "Encl_var": [round(var_enclosu, 2)],
            "Conc_max": [round(max_concent, 2)],
            "Conc_min": [round(min_concent, 2)],
            "Conc_median": [round(median_concent, 2)],
            "Conc_mean": [round(mean_concent, 2)],
            "Conc_std": [round(std_concent, 2)],
            "Conc_var": [round(var_concent, 2)],
            "Overlap_max": [round(max_overlap, 2)],
            "Overlap_min": [round(min_overlap, 2)],
            "Overlap_median": [round(median_overlap, 2)],
            "Overlap_mean": [round(mean_overlap, 2)],
            "Overlap_std": [round(std_overlap, 2)],
            "Overlap_var": [round(var_overlap, 2)],
        }
    )

    results.to_csv(file_name, mode="a", header=not file_exists, index=False)


def heatmap_and_histogram_vector(
    x_labels, y_labels, matrix, matrix_name, ref_img, conf2
):
    x_labels = [label[46:] for label in x_labels]
    y_labels = [label[46:] for label in y_labels]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Overlap (%)", rotation=-90, va="bottom")

    ax.set_xlim(0, 100)

    ax.set_xticks(range(len(y_labels)), labels=y_labels, rotation=90, ha="right")
    ax.set_yticks(range(len(x_labels)), labels=x_labels)

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(
                j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="w"
            )

    title_base = f"{ref_img} vs {conf2} - {matrix_name}"
    ax.set_title(title_base)
    fig.tight_layout()
    plt.savefig(f"../overlap_results_plots/{title_base}_Heatmap.png")

    flat_data = matrix.flatten()
    flat_data = flat_data[flat_data > 0]

    plt.figure()
    plt.hist(flat_data, bins=range(0, 101, 5))
    plt.title(f"{ref_img} vs {conf2} - {matrix_name} Histogram")
    plt.xlabel("Overlap (%)")
    plt.ylabel("Frequency")
    plt.savefig(f"../overlap_results_plots/{title_base}_Histogram.png")
    plt.close("all")


def heatmap_and_histogram_matrix(x_labels, y_labels, matrix, matrix_name, conf1, conf2):
    x_labels = [label[46:] for label in x_labels]
    y_labels = [label[46:] for label in y_labels]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Overlap (%)", rotation=-90, va="bottom")

    ax.set_xticks(
        range(len(y_labels)),
        labels=y_labels,
        rotation=90,
        ha="right",
        rotation_mode="anchor",
    )

    # ax.set_xlim(0, 100)
    ax.set_yticks(range(len(x_labels)), labels=x_labels)

    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(
                j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="w"
            )

    chm_matrix_title = f"{conf1} vs {conf2} - {matrix_name} Heatmap"
    chm_histogram_title = f"{conf1} vs {conf2} - {matrix_name} Histogram"

    ax.set_title(chm_matrix_title)
    fig.tight_layout()
    plt.savefig(f"../overlap_results_plots/{chm_matrix_title}.png")

    hist_no_diagonal = matrix_without_diagonal(matrix)

    plt.figure()
    plt.hist(hist_no_diagonal, bins=range(0, 101, 5))
    plt.title(f"{matrix_name} Histogram (Off-Diagonal) - {conf1} vs {conf2}")
    plt.xlabel("Overlap (%)")
    plt.ylabel("Frequency")
    plt.savefig(f"../overlap_results_plots/{chm_histogram_title}.png")
    plt.close("all")

    """
    similarity_score_th = 40.0
    n_images_above_th = np.sum(hist_no_diagonal > similarity_score_th)
    print(
        f"{matrix_name} - {conf1} vs {conf2}: Number of image pairs with score above {similarity_score_th}%: {n_images_above_th}/{len(hist_no_diagonal)}"
    )
    """

def plot_matrix_solution(manager, routing, solution, distance_matrix, save_path):
    N = distance_matrix.shape[0]

    # Extract route edges and sequence
    index = routing.Start(0)
    route_edges = []
    while not routing.IsEnd(index):
        prev_node = manager.IndexToNode(index)
        index = solution.Value(routing.NextVar(index))
        curr_node = manager.IndexToNode(index)
        route_edges.append((prev_node, curr_node))

    # Set of path coordinates to exclude from general printing
    path_coords = set(route_edges)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the matrix values
    ax.matshow(distance_matrix, cmap="Blues", alpha=0.3)

    for i in range(N):
        for j in range(N):
            # Skip if this cell is part of the route (will be drawn in next loop)
            if (i, j) in path_coords:
                continue
            val = distance_matrix[i][j]
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=8)

    # Highlight path and draw sequence arrows
    prev_pos = None

    for step, (u, v) in enumerate(route_edges):
        # Cell coordinates: row=u, col=v -> (x=v, y=u)

        # Highlight cell in green
        rect = plt.Rectangle(
            (v - 0.5, u - 0.5),
            1,
            1,
            fill=True,
            facecolor="lightgreen",
            edgecolor="green",
            lw=2,
        )
        ax.add_patch(rect)

        # Add step number
        ax.text(
            v,
            u - 0.25,
            f"#{step + 1}",
            ha="center",
            va="center",
            color="darkgreen",
            fontsize=12,
            fontweight="bold",
        )

        # Re-draw the distance value on top of the highlight so it's visible
        ax.text(
            v,
            u + 0.15,
            str(distance_matrix[u][v]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

        # Draw red arrows connecting the steps in the matrix
        curr_pos = (v, u)
        if prev_pos is not None:
            ax.annotate(
                "",
                xy=curr_pos,
                xytext=prev_pos,
                arrowprops=dict(
                    arrowstyle="->", color="red", lw=2, connectionstyle="arc3,rad=0.1"
                ),
            )

        prev_pos = curr_pos

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xlabel("To Node")
    ax.set_ylabel("From Node")
    # ax.set_title("Distance Matrix\n(1 - Overlap)\nwith TSP Path")
    plt.savefig(save_path)
    # plt.show()
    plt.close()
