from matplotlib import pyplot as plt
from functools import reduce
import numpy as np

from .file_utilities import *

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def plot_ibm_network_inference(
    data_dir,
    device_name,
    shots_list,
    num_qubits,
    cov_mat_match=[],
    mi_char_mat_match=[],
    mmi_char_mat_match=[],
):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)

    for i, shots in enumerate(shots_list):
        shots_path = "shots_" + str(shots)

        filenames = get_files(
            data_dir + shots_path, device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
        )
        print("filenmaens , ", filenames)
        data_jsons = list(map(read_json, filenames))
        num_iterations = min([len(data_json["cov_mats"]) for data_json in data_jsons])
        num_trials = len(data_jsons)

        """
        plotting covariance matrix data
        """

        cov_mats_data = [
            [
                np.abs(cov_mat_match - np.abs(np.array(data_jsons[i]["cov_mats"][j])))
                for i in range(num_trials)
            ]
            for j in range(num_iterations)
        ]

        # mean and min distance from optimal covariance matrix
        mean_cov_mats_data = [sum(cov_mats_data[i]) / num_trials for i in range(num_iterations)]
        min_cov_mats_data = [
            reduce(lambda x_mat, y_mat: np.minimum(x_mat, y_mat), cov_mats)
            for cov_mats in cov_mats_data
        ]

        mean_qubit_distances = np.array(
            [
                (np.sum(mean_cov_mat) + np.trace(mean_cov_mat)) / (num_qubits + num_qubits**2)
                for mean_cov_mat in mean_cov_mats_data
            ]
        )
        mean_min_qubit_distances = np.array(
            [
                (np.sum(min_cov_mat) + np.trace(min_cov_mat)) / (num_qubits + num_qubits**2)
                for min_cov_mat in min_cov_mats_data
            ]
        )

        mean_min_std_err = np.array(
            [np.std(min_cov_mat) / np.sqrt(25) for min_cov_mat in min_cov_mats_data]
        )
        mean_dist_std_err = np.array(
            [np.std(mean_cov_mat) / np.sqrt(25) for mean_cov_mat in mean_cov_mats_data]
        )

        ax1.semilogy(
            range(num_iterations),
            mean_qubit_distances,
            label="shots " + str(shots),
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax1.fill_between(
            range(num_iterations),
            mean_qubit_distances - mean_dist_std_err,
            mean_qubit_distances + mean_dist_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        ax1.semilogy(
            range(num_iterations),
            mean_min_qubit_distances,
            linestyle="--",
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax1.fill_between(
            range(num_iterations),
            mean_min_qubit_distances - mean_min_std_err,
            mean_min_qubit_distances + mean_min_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        """
        plotting von neumann entropy data
        """
        vn_ent_match = np.diag(mi_char_mat_match)
        vn_ents_data = [
            [np.abs(vn_ent_match - data_jsons[i]["vn_entropies"][j]) for i in range(num_trials)]
            for j in range(num_iterations)
        ]

        mean_vn_ents_data = [sum(vn_ents_data[i]) / num_trials for i in range(num_iterations)]
        min_vn_dist_data = [
            reduce(lambda x, y: np.minimum(x, y), vn_ents) for vn_ents in vn_ents_data
        ]

        mean_vn_qubit_distances = np.array(
            [np.sum(mean_vn_ents) / num_qubits for mean_vn_ents in mean_vn_ents_data]
        )
        mean_min_vn_qubit_distances = np.array(
            [np.sum(min_vn_dists) / num_qubits for min_vn_dists in min_vn_dist_data]
        )

        min_vn_qubit_distances_std_err = np.array(
            [np.std(min_vn_ents) / num_qubits for min_vn_ents in min_vn_dist_data]
        )
        mean_vn_qubit_distances_std_err = np.array(
            [np.std(mean_vn_ents) / num_qubits for mean_vn_ents in mean_vn_ents_data]
        )

        ax2.semilogy(
            range(num_iterations),
            mean_vn_qubit_distances,
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax2.fill_between(
            range(num_iterations),
            mean_vn_qubit_distances - mean_vn_qubit_distances_std_err,
            mean_vn_qubit_distances + mean_vn_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        ax2.semilogy(
            range(num_iterations),
            mean_min_vn_qubit_distances,
            linestyle="--",
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax2.fill_between(
            range(num_iterations),
            mean_min_vn_qubit_distances - min_vn_qubit_distances_std_err,
            mean_min_vn_qubit_distances + min_vn_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        """
        plotting classical mutual info entropy data
        """
        mi_match = []
        for q1 in range(num_qubits):
            for q2 in range(q1 + 1, num_qubits):
                mi_match += [mi_char_mat_match[q1, q2]]
        mi_match = np.array(mi_match)

        mi_data = [
            [np.abs(mi_match - data_jsons[i]["mutual_infos"][j]) for i in range(num_trials)]
            for j in range(num_iterations)
        ]

        mean_mi_data = [sum(mi_data[i]) / num_trials for i in range(num_iterations)]
        min_mi_data = [
            reduce(lambda x, y: np.minimum(x, y), mutual_infos) for mutual_infos in mi_data
        ]

        mean_mi_qubit_distances = np.array(
            [np.sum(mean_mutual_infos) / num_qubits for mean_mutual_infos in mean_mi_data]
        )
        mean_min_mi_qubit_distances = np.array(
            [np.sum(min_mutual_infos) / num_qubits for min_mutual_infos in min_mi_data]
        )

        min_mi_qubit_distances_std_err = np.array(
            [np.std(min_mutual_infos) / num_qubits for min_mutual_infos in min_mi_data]
        )
        mean_mi_qubit_distances_std_err = np.array(
            [np.std(mean_mutual_infos) / num_qubits for mean_mutual_infos in mean_mi_data]
        )

        ax3.semilogy(
            range(num_iterations),
            mean_mi_qubit_distances,
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax3.fill_between(
            range(num_iterations),
            mean_mi_qubit_distances - mean_mi_qubit_distances_std_err,
            mean_mi_qubit_distances + mean_mi_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        ax3.semilogy(
            range(num_iterations),
            mean_min_mi_qubit_distances,
            linestyle="--",
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax3.fill_between(
            range(num_iterations),
            mean_min_mi_qubit_distances - min_mi_qubit_distances_std_err,
            mean_min_mi_qubit_distances + min_mi_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        """
        plotting meaasured mutual info entropy data
        """
        mmi_match = []
        for q1 in range(num_qubits):
            for q2 in range(q1 + 1, num_qubits):
                mmi_match += [mmi_char_mat_match[q1, q2]]
        mmi_match = np.array(mmi_match)

        mmi_data = [
            [
                np.abs(mmi_match - data_jsons[i]["measured_mutual_infos"][j])
                for i in range(num_trials)
            ]
            for j in range(num_iterations)
        ]

        mean_mmi_data = [sum(mmi_data[i]) / num_trials for i in range(num_iterations)]
        min_mmi_data = [
            reduce(lambda x, y: np.minimum(x, y), mutual_infos) for mutual_infos in mmi_data
        ]

        mean_mmi_qubit_distances = np.array(
            [np.sum(mean_mutual_infos) / num_qubits for mean_mutual_infos in mean_mmi_data]
        )
        mean_min_mmi_qubit_distances = np.array(
            [np.sum(min_mutual_infos) / num_qubits for min_mutual_infos in min_mmi_data]
        )

        min_mmi_qubit_distances_std_err = np.array(
            [np.std(min_mutual_infos) / num_qubits for min_mutual_infos in min_mmi_data]
        )
        mean_mmi_qubit_distances_std_err = np.array(
            [np.std(mean_mutual_infos) / num_qubits for mean_mutual_infos in mean_mmi_data]
        )

        ax4.semilogy(
            range(num_iterations),
            mean_mmi_qubit_distances,
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax4.fill_between(
            range(num_iterations),
            mean_mmi_qubit_distances - mean_mmi_qubit_distances_std_err,
            mean_mmi_qubit_distances + mean_mmi_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

        ax4.semilogy(
            range(num_iterations),
            mean_min_mmi_qubit_distances,
            linestyle="--",
            color=_COLORS[i],
            alpha=3 / 4,
        )
        ax4.fill_between(
            range(num_iterations),
            mean_min_mmi_qubit_distances - min_mmi_qubit_distances_std_err,
            mean_min_mmi_qubit_distances + min_mmi_qubit_distances_std_err,
            alpha=1 / 4,
            color=_COLORS[i],
        )

    fig.legend()

    plt.show()
