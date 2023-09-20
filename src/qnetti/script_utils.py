from time import time


from .file_utilities import tmp_dir, mkdir, write_json, datetime_now_string
from .covariance_matrices import *
from .characteristic_matrices import *
from .qnodes import qubit_probs_qnode_fn


def infer_ibm_network_shot_dependence(
    provider,
    prep_node,
    ibm_device_name="ibmq_qasm_simulator",
    shots_list=[10, 100, 1000, 10000],
    meas_wires=None,
    prep_node_name="",
    num_cov_steps=0,
    num_vn_steps=0,
    num_mi_steps=0,
    num_mmi_steps=0,
    mi_step_size=0.1,
    mmi_step_size=0.25,
    cov_step_size=0.1,
    vn_step_size=0.1,
    cov_init_json={},
    vn_init_json={},
    mi_init_json={},
    mmi_init_json={},
    init_data_json={},
    warm_start_step=0,
):
    """
    Performs network inference on an IBMQ machine over a range of shot numbers.
    The prepared state is specified as the ``prep_node`` and the number of shots
    are passed as the ``shots_list`` parameter.


    The connection to the IBM hardware requires an IBMQ account. The IBM provider can
    be constructed using the private ``token`` as:

    .. code-block:: python

        token = "XYZ"   # secret IBMQ API token for your account
        IBMQ.save_account(token=token, hub="ibm-q", group="open", project="main", overwrite=True)

        provider = IBMQ.load_account()


    :param provide: An IBM provider (see above).
    :type provider: IBMQ.provide

    :param prep_node: A qNetVO ``PrepareNode`` class describing the state to infer.
    :type prep_node: qnetvo.PrepareNode
    """

    num_qubits = len(meas_wires) if meas_wires else len(prep_node.wires)

    dev_kwargs = (
        {
            "name": "qiskit.ibmq",
            "backend": ibm_device_name,
            "provider": provider,
        }
        if ibm_device_name != "default.qubit"
        else {"name": "default.qubit"}
    )
    qnode_kwargs = {"diff_method": "parameter-shift"}

    filepath = mkdir("./data/", "ibm_inference_" + prep_node_name + "_shot_dependence/")

    data_jsons = []
    for shots_id, shots in enumerate(shots_list):
        shots_filepath = mkdir(filepath, "shots_" + str(shots) + "/")
        tmp_filepath = tmp_dir(shots_filepath)

        if num_cov_steps:
            shots_cov_filepath = mkdir(shots_filepath, "cov_opts/")

        if num_vn_steps:
            shots_vn_filepath = mkdir(shots_filepath, "vn_opts/")

        if num_mi_steps:
            shots_mi_filepath = mkdir(shots_filepath, "mi_opts/")

        if num_mmi_steps:
            shots_mmi_filepath = mkdir(shots_filepath, "mmi_opts/")

        dev_kwargs["shots"] = shots

        # helper functions to obtain per qubit cost data
        cov_mat_fn = qubit_covariance_matrix_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )
        probs_qnode, dev = qubit_probs_qnode_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )
        qubit_mmis = qubit_measured_mutual_infos_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )

        meta_opt_kwargs = {
            "ibm_device": ibm_device_name,
            "shots": shots,
            "prep_node_name": prep_node_name,
            "num_qubits": num_qubits,
            "step_only": True,
        }

        cov_opt_dict = cov_init_json if shots_id == 0 else {}
        vn_opt_dict = vn_init_json if shots_id == 0 else {}
        mi_opt_dict = mi_init_json if shots_id == 0 else {}
        mmi_opt_dict = mmi_init_json if shots_id == 0 else {}

        curr_step = warm_start_step if shots_id == 0 else 0
        num_steps = max(num_cov_steps, num_vn_steps, num_mi_steps, num_mmi_steps)
        file_name = ibm_device_name + "_" + datetime_now_string()

        cov_err = False
        vn_err = False
        mi_err = False
        mmi_err = False

        for step in range(curr_step, num_steps):
            print("num shots :  ", shots, ", step :  ", step, " / ", num_steps - 1)
            step_start_time = time()

            if step < num_cov_steps and not (cov_err):
                cov_opt_kwargs = {
                    "num_steps": len(cov_opt_dict["settings_list"]) if cov_opt_dict else 1,
                    "step_size": cov_step_size,
                    "init_opt_dict": cov_opt_dict,
                    "filepath": shots_cov_filepath,
                    "filename": file_name,
                    **meta_opt_kwargs,
                }

                cov_mat, cov_opt_dict = optimize_covariance_matrix(
                    prep_node,
                    meas_wires=meas_wires,
                    dev_kwargs=dev_kwargs,
                    qnode_kwargs=qnode_kwargs,
                    **cov_opt_kwargs,
                )

                if cov_opt_dict["error"]:
                    cov_err = True
                    print("cov opt error")
                else:
                    if "cov_mats" in cov_opt_dict:
                        cov_opt_dict["cov_mats"] += [cov_mat.tolist()]
                    else:
                        # add inital cov mat and cov mat after first optimization step
                        cov_opt_dict["cov_mats"] = [
                            cov_mat_fn(cov_opt_dict["settings_list"][0]).tolist(),
                            cov_mat.tolist(),
                        ]

                    tmp_path = tmp_dir(shots_cov_filepath)
                    write_json(cov_opt_dict, tmp_path + file_name)

                    print("cov opt step time : ", cov_opt_dict["opt_step_times"][-1])

            if step < num_vn_steps and not (vn_err):
                vn_opt_kwargs = {
                    "num_steps": len(vn_opt_dict["settings_list"]) if vn_opt_dict else 1,
                    "step_size": vn_step_size,
                    "init_opt_dict": vn_opt_dict,
                    "filepath": shots_vn_filepath,
                    "filename": file_name,
                    **meta_opt_kwargs,
                }

                vn_opt_dict = optimize_vn_entropy(
                    prep_node,
                    meas_wires=meas_wires,
                    dev_kwargs=dev_kwargs,
                    qnode_kwargs=qnode_kwargs,
                    **vn_opt_kwargs,
                )

                if vn_opt_dict["error"]:
                    vn_err = True
                    print("vn opt error")
                else:
                    vn_entropies = qubit_shannon_entropies(
                        probs_qnode(vn_opt_dict["settings_list"][-1])
                    )
                    if "vn_entropies" in vn_opt_dict:
                        vn_opt_dict["vn_entropies"] += [vn_entropies]
                    else:
                        vn_opt_dict["vn_entropies"] = [
                            qubit_shannon_entropies(probs_qnode(vn_opt_dict["settings_list"][0])),
                            vn_entropies,
                        ]

                    tmp_path = tmp_dir(shots_vn_filepath)
                    write_json(vn_opt_dict, tmp_path + file_name)

                    print("vn opt step time : ", vn_opt_dict["opt_step_times"][-1])

            if step < num_mi_steps and not (mi_err):
                mi_opt_kwargs = {
                    "num_steps": len(mi_opt_dict["settings_list"]) if mi_opt_dict else 1,
                    "step_size": mi_step_size,
                    "init_opt_dict": mi_opt_dict,
                    "filepath": shots_mi_filepath,
                    "filename": file_name,
                    **meta_opt_kwargs,
                }

                mi_opt_dict = optimize_mutual_info(
                    prep_node,
                    meas_wires=meas_wires,
                    dev_kwargs=dev_kwargs,
                    qnode_kwargs=qnode_kwargs,
                    **mi_opt_kwargs,
                )

                if mi_opt_dict["error"]:
                    mi_err = True
                    print("mi opt error")
                else:
                    mutual_infos = qubit_mutual_infos(probs_qnode(mi_opt_dict["settings_list"][-1]))
                    if "mutual_infos" in mi_opt_dict:
                        mi_opt_dict["mutual_infos"] += [mutual_infos]
                    else:
                        mi_opt_dict["mutual_infos"] = [
                            qubit_mutual_infos(probs_qnode(mi_opt_dict["settings_list"][0])),
                            mutual_infos,
                        ]

                    tmp_path = tmp_dir(shots_mi_filepath)
                    write_json(mi_opt_dict, tmp_path + file_name)

                    print("mi opt step time : ", mi_opt_dict["opt_step_times"][-1])

            if step < num_mmi_steps and not (mmi_err):
                mmi_opt_kwargs = {
                    "num_steps": len(mmi_opt_dict["settings_list"]) if mmi_opt_dict else 1,
                    "step_size": mmi_step_size,
                    "init_opt_dict": mmi_opt_dict,
                    "filepath": shots_mmi_filepath,
                    "filename": file_name,
                    **meta_opt_kwargs,
                }
                mmi_opt_dict = optimize_measured_mutual_info(
                    prep_node,
                    meas_wires=meas_wires,
                    dev_kwargs=dev_kwargs,
                    qnode_kwargs=qnode_kwargs,
                    **mmi_opt_kwargs,
                )

                if mmi_opt_dict["error"]:
                    mmi_err = True
                    print("mmi opt error")
                else:
                    measured_mutual_infos = qubit_mmis(mmi_opt_dict["settings_list"][-1])
                    if "measured_mutual_infos" in mmi_opt_dict:
                        mmi_opt_dict["measured_mutual_infos"] += [measured_mutual_infos]
                    else:
                        mmi_opt_dict["measured_mutual_infos"] = [
                            qubit_mmis(mmi_opt_dict["settings_list"][0]),
                            measured_mutual_infos,
                        ]

                    tmp_path = tmp_dir(shots_mmi_filepath)
                    write_json(mmi_opt_dict, tmp_path + file_name)

                    print("mmi opt step time : ", mmi_opt_dict["opt_step_times"][-1])

                    print("Iteration time : ", time() - step_start_time)

        # write data after complete optimizataion
        if num_cov_steps and not (cov_err):
            write_json(
                cov_opt_dict,
                shots_cov_filepath + file_name,
            )

        if num_vn_steps and not (vn_err):
            write_json(
                vn_opt_dict,
                shots_vn_filepath + file_name,
            )

        if num_mi_steps and not (mi_err):
            write_json(
                mi_opt_dict,
                shots_mi_filepath + file_name,
            )

        if num_mmi_steps and not (mmi_err):
            write_json(
                mmi_opt_dict,
                shots_mmi_filepath + file_name,
            )
