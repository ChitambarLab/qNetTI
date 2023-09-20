from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from functools import reduce
import numpy as np
import math
import qnetvo

from .file_utilities import *
from .characteristic_matrices import *
from .covariance_matrices import *
from .qnodes import qubit_probs_qnode_fn

_COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
_MARKERS = ["o", "d", "P", "^"]


def plot_ibm_network_inference(
    data_dir,
    shots_list,
    num_qubits,
    prep_node,
    ibm_device_name="ibmq_belem",
    sim_device_name="default.qubit",
    title="",
    cov_mat_match=[],
    mi_char_mat_match=[],
    mmi_char_mat_match=[],
    opt_yticks=[],
    avg_data=None,
):
    """
    Creates a plot of the data acquired from the quantum computing hardware.
    See Fig. 8 of our `Arxiv Paper <https://arxiv.org/abs/2212.07987>`_.
    """
    # setting font sizes
    title_fontsize = 20
    subplot_fontsize = 12

    # configuring figures and axes
    opt_fig, (
        (cov_opt_ax1, char_opt_ax1, vn_opt_ax1, mi_opt_ax1),
        (cov_opt_ax2, char_opt_ax2, vn_opt_ax2, mi_opt_ax2),
        (cov_opt_ax3, char_opt_ax3, vn_opt_ax3, mi_opt_ax3),
    ) = plt.subplots(ncols=4, nrows=3, figsize=(14, 8), constrained_layout=True)

    opt_fig.suptitle(title + " Qubit Topology Inference Error", fontsize=title_fontsize)

    cov_opt_ax1.set_title("Covariance", fontweight="bold")
    char_opt_ax1.set_title("Characteristic", fontweight="bold")
    vn_opt_ax1.set_title("Shannon Entropy", fontweight="bold")
    mi_opt_ax1.set_title("Mutual Information", fontweight="bold")

    cov_opt_ax3.set_xlabel("Optimization Step", fontsize=subplot_fontsize)
    char_opt_ax3.set_xlabel("Optimization Step", fontsize=subplot_fontsize)
    vn_opt_ax3.set_xlabel("Optimization Step", fontsize=subplot_fontsize)
    mi_opt_ax3.set_xlabel("Optimization Step", fontsize=subplot_fontsize)

    cov_opt_ax1.set_ylabel("Error", fontsize=subplot_fontsize)
    cov_opt_ax2.set_ylabel("Error", fontsize=subplot_fontsize)
    cov_opt_ax3.set_ylabel("Error", fontsize=subplot_fontsize)

    cov_opt_ax1.grid()
    cov_opt_ax2.grid()
    cov_opt_ax3.grid()
    char_opt_ax1.grid()
    char_opt_ax2.grid()
    char_opt_ax3.grid()
    vn_opt_ax1.grid()
    vn_opt_ax2.grid()
    vn_opt_ax3.grid()
    mi_opt_ax1.grid()
    mi_opt_ax2.grid()
    mi_opt_ax3.grid()

    for ax_title, ax in [
        ("Classical\nSimulator", cov_opt_ax1),
        ("IBM Hardware\n(Noisy)", cov_opt_ax2),
        ("IBM Hardware\n(Noiseless)", cov_opt_ax3),
    ]:
        ax.annotate(
            ax_title,
            xy=(-1.5, 0.5),
            xytext=(-ax.yaxis.labelpad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size=subplot_fontsize,
            ha="center",
            va="center",
            rotation=90,
            fontweight="bold",
        )

    # functions for collecting noiseless data from optimized settings
    cov_mat_fn = qubit_covariance_matrix_fn(prep_node, meas_wires=range(num_qubits))
    probs_qnode, dev = qubit_probs_qnode_fn(prep_node, meas_wires=range(num_qubits))

    line_handles = []

    for i, shots in enumerate(shots_list):
        shots_path = "shots_" + str(shots)

        mean_line_styles = {
            "color": _COLORS[i],
            "linestyle": "-",
            "marker": _MARKERS[i],
            "alpha": 3 / 4,
        }

        std_err_line_styles = {
            "color": _COLORS[i],
            "linestyle": "-",
            "alpha": 1 / 8,
        }

        for device_name in [ibm_device_name, sim_device_name]:
            """
            plotting covariance matrix data
            """
            cov_shots_path = shots_path + "/cov_opts"
            cov_filenames = get_files(
                data_dir + cov_shots_path, device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
            )
            cov_data_jsons = list(map(read_json, cov_filenames))
            num_cov_trials = len(cov_data_jsons)
            num_cov_iterations = len(cov_data_jsons[-1]["cov_mats"])

            print("Cov, shots ", shots)
            print("settings list len ", len(cov_data_jsons[0]["settings_list"]))
            print("cov mats len ", len(cov_data_jsons[0]["cov_mats"]))

            cov_mats_data = [
                [
                    # np.abs(cov_mat_match - np.abs(np.array(cov_data_jsons[k]["cov_mats"][j])))
                    np.abs(np.array(cov_data_jsons[k]["cov_mats"][j]))
                    for k in range(num_cov_trials)
                ]
                for j in range(num_cov_iterations)
            ]

            mean_cov_dists, mean_cov_dist_std_err = _mean_mat_dists(cov_mats_data, cov_mat_match)

            if device_name == sim_device_name:
                (cov_line_i,) = cov_opt_ax1.semilogy(
                    range(num_cov_iterations),
                    mean_cov_dists,
                    **mean_line_styles,
                    label="shots " + str(shots),
                )
                cov_opt_ax1.fill_between(
                    range(num_cov_iterations),
                    mean_cov_dists - mean_cov_dist_std_err,
                    mean_cov_dists + mean_cov_dist_std_err,
                    **std_err_line_styles,
                )

                line_handles += [cov_line_i]
            else:
                # compute noiseless data for hardware results
                # compute inference heat maps for noisy hardware results
                for opt_dict in cov_data_jsons:
                    opt_dict["noiseless_cov_mats"] = []
                    for settings in opt_dict["settings_list"]:
                        opt_dict["noiseless_cov_mats"] += [cov_mat_fn(settings)]

                noiseless_cov_mats_data = [
                    [
                        # np.abs(cov_mat_match - np.abs(np.array(cov_data_jsons[k]["noiseless_cov_mats"][j])))
                        np.abs(np.array(cov_data_jsons[k]["noiseless_cov_mats"][j]))
                        for k in range(num_cov_trials)
                    ]
                    for j in range(num_cov_iterations)
                ]

                noiseless_mean_cov_dists, noiseless_mean_cov_dist_std_err = _mean_mat_dists(
                    noiseless_cov_mats_data, cov_mat_match
                )

                cov_opt_ax2.semilogy(range(num_cov_iterations), mean_cov_dists, **mean_line_styles)
                cov_opt_ax2.fill_between(
                    range(num_cov_iterations),
                    mean_cov_dists - mean_cov_dist_std_err,
                    mean_cov_dists + mean_cov_dist_std_err,
                    **std_err_line_styles,
                )
                cov_opt_ax3.semilogy(
                    range(num_cov_iterations),
                    noiseless_mean_cov_dists,
                    **mean_line_styles,
                )
                cov_opt_ax3.fill_between(
                    range(num_cov_iterations),
                    noiseless_mean_cov_dists - noiseless_mean_cov_dist_std_err,
                    noiseless_mean_cov_dists + noiseless_mean_cov_dist_std_err,
                    **std_err_line_styles,
                )

            """
            loading von neumann entropy data
            """
            vn_shots_path = shots_path + "/vn_opts"
            vn_filenames = get_files(
                data_dir + vn_shots_path, device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
            )
            vn_data_jsons = list(map(read_json, vn_filenames))

            print("VN, shots ", shots)
            print("settings list len ", len(vn_data_jsons[0]["settings_list"]))
            print("vn mats len ", len(vn_data_jsons[0]["vn_entropies"]))

            """
            Loading mutual info data
            """
            mi_shots_path = shots_path + "/mi_opts"
            mi_filenames = get_files(
                data_dir + mi_shots_path, device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
            )
            mi_data_jsons = list(map(read_json, mi_filenames))

            print("MI, shots ", shots)
            print("settings list len ", len(mi_data_jsons[0]["settings_list"]))
            print("mi mats len ", len(mi_data_jsons[0]["mutual_infos"]))

            """
            constructing/plotting characteristic matrices
            """
            char_mats, shannon_ents, mutual_infos = _char_mats_from_data_jsons(
                vn_data_jsons, mi_data_jsons, avg_data=avg_data
            )
            mean_char_dists, mean_char_dist_std_err = _mean_mat_dists(char_mats, mi_char_mat_match)

            vn_match = np.diag(mi_char_mat_match)
            mi_match = []
            for q1 in range(len(vn_match)):
                for q2 in range(q1 + 1, len(vn_match)):
                    mi_match += [mi_char_mat_match[q1, q2]]
            mi_match = np.array(mi_match)

            mean_vn_dists, mean_vn_dist_std_err = _mean_mat_dists(shannon_ents, vn_match)
            mean_mi_dists, mean_mi_dist_std_err = _mean_mat_dists(mutual_infos, mi_match)

            num_char_iterations = len(mean_char_dists)
            if device_name == sim_device_name:
                char_opt_ax1.semilogy(
                    range(num_char_iterations),
                    mean_char_dists,
                    **mean_line_styles,
                    label="shots " + str(shots),
                )
                char_opt_ax1.fill_between(
                    range(num_char_iterations),
                    mean_char_dists - mean_char_dist_std_err,
                    mean_char_dists + mean_char_dist_std_err,
                    **std_err_line_styles,
                )
                vn_opt_ax1.semilogy(
                    range(len(mean_vn_dists)),
                    mean_vn_dists,
                    **mean_line_styles,
                    label="shots " + str(shots),
                )
                vn_opt_ax1.fill_between(
                    range(len(mean_vn_dists)),
                    mean_vn_dists - mean_vn_dist_std_err,
                    mean_vn_dists + mean_vn_dist_std_err,
                    **std_err_line_styles,
                )
                mi_opt_ax1.semilogy(
                    range(len(mean_mi_dists)),
                    mean_mi_dists,
                    **mean_line_styles,
                    label="shots " + str(shots),
                )
                mi_opt_ax1.fill_between(
                    range(len(mean_mi_dists)),
                    mean_mi_dists - mean_mi_dist_std_err,
                    mean_mi_dists + mean_mi_dist_std_err,
                    **std_err_line_styles,
                )
            else:
                for opt_dict in vn_data_jsons:
                    opt_dict["noiseless_vn_entropies"] = []
                    for settings in opt_dict["settings_list"]:
                        opt_dict["noiseless_vn_entropies"] += [
                            qubit_shannon_entropies(probs_qnode(settings))
                        ]

                for opt_dict in mi_data_jsons:
                    opt_dict["noiseless_mutual_infos"] = []
                    for settings in opt_dict["settings_list"]:
                        opt_dict["noiseless_mutual_infos"] += [
                            qubit_mutual_infos(probs_qnode(settings))
                        ]

                (
                    noiseless_char_mats,
                    noiseless_shannon_ents,
                    noiseless_mutual_infos,
                ) = _char_mats_from_data_jsons(
                    vn_data_jsons,
                    mi_data_jsons,
                    noiseless=True,
                    avg_data=avg_data,
                )
                noiseless_mean_char_dists, noiseless_mean_char_dist_std_err = _mean_mat_dists(
                    noiseless_char_mats, mi_char_mat_match
                )
                noiseless_mean_vn_dists, noiseless_mean_vn_dist_std_err = _mean_mat_dists(
                    noiseless_shannon_ents, vn_match
                )
                noiseless_mean_mi_dists, noiseless_mean_mi_dist_std_err = _mean_mat_dists(
                    noiseless_mutual_infos, mi_match
                )

                noiseless_mean_vn_dists, noiseless_mean_vn_dist_std_err = _mean_mat_dists(
                    noiseless_shannon_ents, vn_match
                )
                noiseless_mean_mi_dists, noiseless_mean_mi_dist_std_err = _mean_mat_dists(
                    noiseless_mutual_infos, mi_match
                )

                num_noiseless_char_iterations = len(noiseless_mean_char_dists)
                char_opt_ax2.semilogy(
                    range(num_char_iterations), mean_char_dists, **mean_line_styles
                )
                char_opt_ax2.fill_between(
                    range(num_char_iterations),
                    mean_char_dists - mean_char_dist_std_err,
                    mean_char_dists + mean_char_dist_std_err,
                    **std_err_line_styles,
                )
                char_opt_ax3.semilogy(
                    range(num_noiseless_char_iterations),
                    noiseless_mean_char_dists,
                    **mean_line_styles,
                )
                char_opt_ax3.fill_between(
                    range(num_noiseless_char_iterations),
                    noiseless_mean_char_dists - noiseless_mean_char_dist_std_err,
                    noiseless_mean_char_dists + noiseless_mean_char_dist_std_err,
                    **std_err_line_styles,
                )

                num_noiseless_vn_iterations = len(noiseless_mean_vn_dists)
                num_vn_iterations = len(mean_vn_dists)
                vn_opt_ax2.semilogy(range(num_vn_iterations), mean_vn_dists, **mean_line_styles)
                vn_opt_ax2.fill_between(
                    range(num_vn_iterations),
                    mean_vn_dists - mean_vn_dist_std_err,
                    mean_vn_dists + mean_vn_dist_std_err,
                    **std_err_line_styles,
                )
                vn_opt_ax3.semilogy(
                    range(num_noiseless_vn_iterations),
                    noiseless_mean_vn_dists,
                    **mean_line_styles,
                )
                vn_opt_ax3.fill_between(
                    range(num_noiseless_vn_iterations),
                    noiseless_mean_vn_dists - noiseless_mean_vn_dist_std_err,
                    noiseless_mean_vn_dists + noiseless_mean_vn_dist_std_err,
                    **std_err_line_styles,
                )

                num_noiseless_mi_iterations = len(noiseless_mean_mi_dists)
                num_mi_iterations = len(mean_mi_dists)
                mi_opt_ax2.semilogy(range(num_mi_iterations), mean_mi_dists, **mean_line_styles)
                mi_opt_ax2.fill_between(
                    range(num_mi_iterations),
                    mean_mi_dists - mean_mi_dist_std_err,
                    mean_mi_dists + mean_mi_dist_std_err,
                    **std_err_line_styles,
                )
                mi_opt_ax3.semilogy(
                    range(num_noiseless_mi_iterations),
                    noiseless_mean_mi_dists,
                    **mean_line_styles,
                )
                mi_opt_ax3.fill_between(
                    range(num_noiseless_mi_iterations),
                    noiseless_mean_mi_dists - noiseless_mean_mi_dist_std_err,
                    noiseless_mean_mi_dists + noiseless_mean_mi_dist_std_err,
                    **std_err_line_styles,
                )

    if len(opt_yticks):
        cov_opt_ax1.set_yticks(opt_yticks[0])
        cov_opt_ax2.set_yticks(opt_yticks[1])
        cov_opt_ax3.set_yticks(opt_yticks[2])
        char_opt_ax1.set_yticks(opt_yticks[3])
        char_opt_ax2.set_yticks(opt_yticks[4])
        char_opt_ax3.set_yticks(opt_yticks[5])

    plt.figlegend(handles=line_handles, loc="lower center", ncol=4, fontsize=subplot_fontsize)

    opt_fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.show()


def _char_mats_from_data_jsons(vn_data_jsons, mi_data_jsons, noiseless=False, avg_data=None):
    mi_key = "mutual_infos"
    vn_key = "vn_entropies"
    if noiseless:
        mi_key = "noiseless_" + mi_key
        vn_key = "noiseless_" + vn_key

    num_mi_trials = len(mi_data_jsons)
    num_vn_trials = len(vn_data_jsons)

    print("num trials vn/mi : ", num_vn_trials, num_mi_trials)

    num_mi_iterations = len(mi_data_jsons[-1][mi_key])
    num_vn_iterations = len(vn_data_jsons[0][vn_key])

    print("num_iterations vn/mi : ", num_vn_iterations, num_mi_iterations)
    num_qubits = len(vn_data_jsons[0][vn_key][0])

    shannon_ents = [
        [vn_data_jsons[k][vn_key][j] for k in range(num_vn_trials)]
        for j in range(num_vn_iterations)
    ]

    mutual_infos = [
        [mi_data_jsons[k][mi_key][j] for k in range(num_mi_trials)]
        for j in range(num_mi_iterations)
    ]

    char_mats = []
    if avg_data == None:
        num_iterations = min(num_mi_iterations, num_vn_iterations)
        num_trials = min(num_mi_trials, num_vn_trials)

        for j in range(num_iterations):
            char_mats += [[]]
            for k in range(num_trials):
                char_mat = np.zeros((num_qubits, num_qubits))
                for q in range(num_qubits):
                    char_mat[q, q] = shannon_ents[j][k][q]

                mi_id = 0
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        char_mat[q1, q2] = mutual_infos[j][k][mi_id]
                        char_mat[q2, q1] = mutual_infos[j][k][mi_id]
                        mi_id += 1

                char_mats[j] += [char_mat]

    elif avg_data == "vn":
        total_sum_vn_entropies = np.zeros(num_qubits)
        for k in range(len(vn_data_jsons)):
            for j in range(len(vn_data_jsons[k][vn_key])):
                total_sum_vn_entropies += np.array(vn_data_jsons[k][vn_key][j])
        total_mean_vn_entropies = total_sum_vn_entropies / (num_vn_iterations * num_vn_trials)

        for j in range(num_mi_iterations):
            char_mats += [[]]
            for k in range(num_mi_trials):
                char_mat = np.zeros((num_qubits, num_qubits))
                for q in range(num_qubits):
                    char_mat[q, q] = total_mean_vn_entropies[q]

                mi_id = 0
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        char_mat[q1, q2] = mi_data_jsons[k][mi_key][j][mi_id]
                        char_mat[q2, q1] = mi_data_jsons[k][mi_key][j][mi_id]
                        mi_id += 1

                char_mats[j] += [char_mat]
    elif avg_data == "mi":
        total_sum_mi = np.zeros(math.comb(num_qubits, 2))
        for j in range(num_mi_iterations):
            for k in range(num_mi_trials):
                total_sum_mi += np.array(mi_data_jsons[k][mi_key][j])
        total_mean_mi = total_sum_mi / (num_mi_iterations * num_mi_trials)

        for j in range(num_vn_iterations):
            char_mats += [[]]
            for k in range(num_vn_trials):
                char_mat = np.zeros((num_qubits, num_qubits))
                for q in range(num_qubits):
                    char_mat[q, q] = vn_data_jsons[k][vn_key][j][q]

                mi_id = 0
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        char_mat[q1, q2] = total_mean_mi[mi_id]
                        char_mat[q2, q1] = total_mean_mi[mi_id]

                char_mats[j] += [char_mat]

    return char_mats, shannon_ents, mutual_infos


def _mean_mat_dists(mats, mat_match):
    mean_mat_dist_std_err = []
    mean_mat_dists = []
    for j in range(len(mats)):
        mean_data_list = []
        for k in range(len(mats[j])):
            # mat_diff = mat_match - np.array(mats[j][k])
            # mean_data_list += [np.sqrt(np.trace(mat_diff.T @ mat_diff))]
            mat_diff_squared = (mat_match - np.array(mats[j][k])) ** 2
            mean_data_list += [np.sqrt(np.sum(mat_diff_squared))]

        mean_mat_dist_std_err += [np.std(mean_data_list) / np.sqrt(len(mats[j]))]
        mean_mat_dists += [sum(mean_data_list) / len(mats[j])]

    mean_mat_dists = np.array(mean_mat_dists)
    mean_mat_dist_std_err = np.array(mean_mat_dist_std_err)

    return mean_mat_dists, mean_mat_dist_std_err


def plot_qubit_inference_heat_map(
    data_dir, device_name, title="", cov_mat_match=None, char_mat_match=None
):
    """
    Creates a heatmap plot of the data acquired from the quantum computing hardware.
    See Fig. 9 of our `Arxiv Paper <https://arxiv.org/abs/2212.07987>`_
    """
    heat_map_fig, (hm_cov_mat_axes, hm_char_mat_axes) = plt.subplots(
        ncols=6, nrows=2, figsize=(14, 5)
    )  # , constrained_layout=True)

    title_fontsize = 18
    subplot_fontsize = 16
    heat_map_fig.suptitle(
        "Optimal Qubit Inference Matrices for a " + title + " on IBM Hardware",
        fontsize=title_fontsize,
    )

    hm_cov_mat_axes[0].set_title("Inference Error\n", fontweight="bold")
    hm_cov_mat_axes[1].set_title("Ideal", fontweight="bold")
    hm_cov_mat_axes[2].set_title("10 Shots", fontweight="bold")
    hm_cov_mat_axes[3].set_title("100 Shots", fontweight="bold")
    hm_cov_mat_axes[4].set_title("1000 Shots", fontweight="bold")
    hm_cov_mat_axes[5].set_title("10000 Shots", fontweight="bold")

    hm_char_mat_axes[0].set_xlabel("Number of Shots")
    hm_char_mat_axes[0].set_ylabel("Distance to Ideal")
    hm_cov_mat_axes[0].set_ylabel("Distance to Ideal")
    hm_char_mat_axes[0].set_xticks([0, 1, 2, 3])
    hm_char_mat_axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    hm_cov_mat_axes[0].set_xticks([0, 1, 2, 3])
    hm_cov_mat_axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    hm_char_mat_axes[0].set_xticklabels(["10", "100", "1K", "10K"])
    hm_cov_mat_axes[0].set_xticklabels(["10", "100", "1K", "10K"])

    for ax_title, ax in [
        ("Covariance\n", hm_cov_mat_axes[0]),
        ("Characteristic\n", hm_char_mat_axes[0]),
    ]:
        ax.annotate(
            ax_title,
            xy=(-1, 0.5),
            xytext=(-ax.yaxis.labelpad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size=subplot_fontsize,
            ha="center",
            va="center",
            rotation=90,
            fontweight="bold",
        )

    pcm_cov = hm_cov_mat_axes[1].matshow(cov_mat_match, vmin=0, vmax=1)
    pcm_char = hm_char_mat_axes[1].matshow(char_mat_match, vmin=0, vmax=1)

    cb_ax1 = heat_map_fig.add_axes([0.95, 0.04, 0.01, 0.8])
    cbar = heat_map_fig.colorbar(pcm_cov, cax=cb_ax1)

    for ax in [*hm_cov_mat_axes[1:], *hm_char_mat_axes[1:]]:
        ax.set_xticks([x - 0.5 for x in range(1, 5)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, 5)], minor=True)
        ax.grid(which="minor", ls="-", lw=1, color="black")

    cov_dists = []
    char_dists = []
    for i, shots in enumerate([10, 100, 1000, 10000]):
        shots_path = "/shots_" + str(shots)

        cov_filenames = get_files(
            data_dir + shots_path + "/cov_opts", device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
        )
        cov_data_jsons = list(map(read_json, cov_filenames))
        num_cov_trials = len(cov_data_jsons)
        num_cov_iterations = len(cov_data_jsons[-1]["cov_mats"])

        mi_filenames = get_files(
            data_dir + shots_path + "/mi_opts", device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
        )
        mi_data_jsons = list(map(read_json, mi_filenames))
        num_mi_trials = len(mi_data_jsons)
        num_mi_iterations = len(mi_data_jsons[-1]["mutual_infos"])

        vn_filenames = get_files(
            data_dir + shots_path + "/vn_opts", device_name + "_\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\dZ"
        )
        vn_data_jsons = list(map(read_json, vn_filenames))
        num_vn_trials = len(vn_data_jsons)
        num_vn_iterations = len(vn_data_jsons[-1]["vn_entropies"])

        max_cov_mat = np.zeros((5, 5))
        # for j in range(num_cov_iterations - 5, num_cov_iterations):

        for j in range(num_cov_iterations):
            for k in range(num_cov_trials):
                cov_mat = np.abs(np.array(cov_data_jsons[k]["cov_mats"][j]))
                for q1 in range(5):
                    for q2 in range(5):
                        if cov_mat[q1, q2] > max_cov_mat[q1, q2]:
                            max_cov_mat[q1, q2] = cov_mat[q1, q2]

        pcm = hm_cov_mat_axes[i + 2].matshow(max_cov_mat, vmin=0, vmax=1)

        min_vn_entropies = np.ones(5)
        # for j in range(num_vn_iterations - 5, num_vn_iterations):
        for j in range(num_vn_iterations):
            for k in range(num_vn_trials):
                vn_ents = vn_data_jsons[k]["vn_entropies"][j]
                for q in range(5):
                    if vn_ents[q] < min_vn_entropies[q]:
                        min_vn_entropies[q] = vn_ents[q]

        max_char_mat = np.zeros((5, 5))

        for q in range(5):
            max_char_mat[q, q] = min_vn_entropies[q]

        # for j in range(num_mi_iterations - 5, num_mi_iterations):
        for j in range(num_mi_iterations):
            for k in range(num_mi_trials):
                mis = np.array(mi_data_jsons[k]["mutual_infos"][j])
                mi_id = 0
                for q1 in range(5):
                    for q2 in range(q1 + 1, 5):
                        if mis[mi_id] > max_char_mat[q1, q2]:
                            max_char_mat[q1, q2] = mis[mi_id]
                            max_char_mat[q2, q1] = mis[mi_id]
                        mi_id += 1

        pcm = hm_char_mat_axes[i + 2].matshow(max_char_mat, vmin=0, vmax=1)

        cov_dist = np.sqrt(np.sum((cov_mat_match - max_cov_mat) ** 2))
        cov_dists += [cov_dist]

        char_dist = np.sqrt(np.sum((char_mat_match - max_char_mat) ** 2))
        char_dists += [char_dist]

    hm_cov_mat_axes[0].bar([0, 1, 2, 3], cov_dists)
    hm_char_mat_axes[0].bar([0, 1, 2, 3], char_dists)

    heat_map_fig.tight_layout()
    heat_map_fig.subplots_adjust(right=0.92)
    plt.show()
