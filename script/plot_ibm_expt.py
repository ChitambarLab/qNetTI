import qnetti
import numpy as np

data_dir = "./data/ibm_inference_5-qubit_ghz_state_shot_dependence/"
ibm_device_name = "ibmq_qasm_simulator"
shots_list = [10, 100, 1000, 10000]
num_qubits = 5

mat_match = np.ones((5,5))

qnetti.plot_ibm_network_inference(
    data_dir, ibm_device_name, shots_list, num_qubits,
    cov_mat_match = mat_match,
    mi_char_mat_match = mat_match,
    mmi_char_mat_match = mat_match,
)

# data_dir = "./data/ibm_inference_5-qubit_zero_state_shot_dependence/"
# ibm_device_name = "ibmq_qasm_simulator"
# shots_list = [10, 100, 1000, 10000]
# num_qubits = 5

# qnetti.plot_ibm_network_inference(
#     data_dir, ibm_device_name, shots_list, num_qubits,
#     cov_mat_match = np.eye(num_qubits),
#     mi_char_mat_match = np.zeros((num_qubits,num_qubits)),
#     mmi_char_mat_match = np.zeros((num_qubits,num_qubits)),
# )

# data_dir = "./data/ibm_inference_W_state_2-qubit_ghz_state_shot_dependence/"
# ibm_device_name = "ibmq_qasm_simulator"
# shots_list = [10, 100, 1000, 10000]
# num_qubits = 5

# cov_match = np.array(
#     [
#         [1, 2 / 3, 2 / 3, 0, 0],
#         [2 / 3, 1, 2 / 23, 0, 0],
#         [2 / 3, 2 / 3, 1, 0, 0],
#         [0, 0, 0, 1, 1],
#         [0, 0, 0, 1, 1],
#     ]
# )
# char_match = np.array(
#     [
#         [1, 0.35, 0.35, 0, 0],
#         [0.35, 1, 0.35, 0, 0],
#         [0.35, 0.35, 1, 0, 0],
#         [0, 0, 0, 1, 1],
#         [0, 0, 0, 1, 1],
#     ]
# )

# qnetti.plot_ibm_network_inference(
#     data_dir,
#     ibm_device_name,
#     shots_list,
#     num_qubits,
#     cov_mat_match=cov_match,
#     mi_char_mat_match=char_match,
#     mmi_char_mat_match=char_match,
# )
