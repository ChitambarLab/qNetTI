import qnetvo
import pennylane_qiskit


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
    num_steps=20,
    mi_step_size=0.1,
    mmi_step_size=0.1,
    cov_step_size=0.1,
    vn_step_size=0.1,
    init_data_json={},
):
    """
    Performs network inference on an IBMQ machine over a range of shot numbers.
    The prepared state is specified as the ``prep_node`` and the number of shots
    are passed aas the ``shots_list`` parameter.
    """

    num_qubits = len(meas_wires) if meas_wires else len(prep_node.wires)

    dev_kwargs = {
        "name": "qiskit.ibmq",
        "backend": ibm_device_name,
        "provider": provider,
    }

    qnode_kwargs = {"diff_method": "parameter-shift"}

    filepath = mkdir("./data/", "ibm_inference_" + prep_node_name + "_shot_dependence/")

    data_jsons = []
    for shots_id, shots in enumerate(shots_list):
        shots_filepath = mkdir(filepath, "shots_" + str(shots) + "/")
        tmp_filepath = tmp_dir(shots_filepath)

        dev_kwargs["shots"] = shots

        # helper functions to obtain per qubit cost data
        probs_qnode, dev = qubit_probs_qnode_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )
        qubit_mmis = qubit_measured_mutual_infos_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )
        cov_mat = qubit_covariance_matrix_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
        )

        data_json = (
            init_data_json
            if init_data_json and shots_id == 0
            else {
                "mmi_opt_dict": {},
                "vn_opt_dict": {},
                "mi_opt_dict": {},
                "cov_opt_dict": {},
                "ibm_device": ibm_device_name,
                "num_steps": num_steps,
                "shots": shots,
                "prep_node_name": prep_node_name,
                "num_qubits": num_qubits,
                "measured_mutual_infos": [],
                "mutual_infos": [],
                "cov_mats": [],
                "vn_entropies": [],
            }
        )

        curr_step = len(data_json["vn_entropies"])

        for step in range(curr_step, num_steps):
            num_opt_steps = (
                len(data_json["cov_opt_dict"]["settings_list"]) if data_json["cov_opt_dict"] else 1
            )

            mi_opt_kwargs = {
                "num_steps": num_opt_steps,
                "step_size": mi_step_size,
                "init_opt_dict": data_json["mi_opt_dict"],
                "filepath": shots_filepath,
                "filename": "mi_opt_dict",
                "step_only": True,
            }

            mmi_opt_kwargs = {
                "num_steps": num_opt_steps,
                "step_size": mmi_step_size,
                "init_opt_dict": data_json["mmi_opt_dict"],
                "filepath": shots_filepath,
                "filename": "mmi_opt_dict",
                "step_only": True,
            }

            vn_opt_kwargs = {
                "num_steps": num_opt_steps,
                "step_size": vn_step_size,
                "init_opt_dict": data_json["vn_opt_dict"],
                "filepath": shots_filepath,
                "filename": "vn_opt_dict",
                "step_only": True,
            }

            cov_opt_kwargs = {
                "num_steps": num_opt_steps,
                "step_size": cov_step_size,
                "init_opt_dict": data_json["cov_opt_dict"],
                "filepath": shots_filepath,
                "filename": "cov_opt_dict",
                "step_only": True,
            }

            datetime_str = datetime_now_string()

            cov_mat, data_json["cov_opt_dict"] = optimize_covariance_matrix(
                prep_node,
                meas_wires=meas_wires,
                dev_kwargs=dev_kwargs,
                qnode_kwargs=qnode_kwargs,
                **cov_opt_kwargs,
            )
            data_json["cov_mats"] += [cov_mat.tolist()]

            data_json["vn_opt_dict"] = optimize_vn_entropy(
                prep_node,
                meas_wires=meas_wires,
                dev_kwargs=dev_kwargs,
                qnode_kwargs=qnode_kwargs,
                **vn_opt_kwargs,
            )
            data_json["vn_entropies"] += [
                qubit_shannon_entropies(probs_qnode(data_json["vn_opt_dict"]["settings_list"][-1]))
            ]

            data_json["mi_opt_dict"] = optimize_mutual_info(
                prep_node,
                meas_wires=meas_wires,
                dev_kwargs=dev_kwargs,
                qnode_kwargs=qnode_kwargs,
                **mi_opt_kwargs,
            )
            data_json["mutual_infos"] += [
                qubit_mutual_infos(probs_qnode(data_json["mi_opt_dict"]["settings_list"][-1]))
            ]

            data_json["mmi_opt_dict"] = optimize_measured_mutual_info(
                prep_node,
                meas_wires=meas_wires,
                dev_kwargs=dev_kwargs,
                qnode_kwargs=qnode_kwargs,
                **mmi_opt_kwargs,
            )
            data_json["measured_mutual_infos"] += [
                qubit_mmis(data_json["mmi_opt_dict"]["settings_list"][-1])
            ]

            # write data after each step
            print(data_json)
            write_json(
                data_json,
                tmp_filepath + ibm_device_name + "_step_" + str(step) + "_" + datetime_str,
            )

        # write data after complete optimizataion
        write_json(
            data_json,
            shots_filepath + ibm_device_name + "_" + datetime_str,
        )

        data_jsons += data_json

    return data_jsons
