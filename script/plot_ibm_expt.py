import qnetti
import qnetvo
import numpy as np

data_dir = "./data/ibm_inference_5-qubit_ghz_state_shot_dependence/"
ibm_device_name = "ibmq_belem"
# ibm_device_name = "default.qubit"
shots_list = [10, 100, 1000, 10000]
num_qubits = 5

mat_match = np.ones((5, 5))

qnetti.plot_qubit_inference_heat_map(
    data_dir,
    ibm_device_name,
    title="5-Qubit GHZ State",
    cov_mat_match=mat_match,
    char_mat_match=mat_match,
)

qnetti.plot_ibm_network_inference(
    data_dir,
    ibm_device_name,
    shots_list,
    num_qubits,
    prep_node=qnetvo.PrepareNode(wires=[0, 1, 2, 3, 4], ansatz_fn=qnetvo.ghz_state),
    title="5-Qubit GHZ State",
    cov_mat_match=mat_match,
    mi_char_mat_match=mat_match,
    mmi_char_mat_match=mat_match,
    # opt_yticks=[
    #     [5,1,5e-1,1e-1,5e-2,1e-2],
    #     [5,1,5e-1],
    #     [5,1,5e-1,1e-1,5e-2],
    #     [5,1,5e-1],
    #     [10,5,1],
    #     [10,5,1,5e-1,1e-1],
    # ],
    avg_data="vn",
)

data_dir = "./data/ibm_inference_5-qubit_zero_state_shot_dependence/"
ibm_device_name = "ibmq_belem"
# ibm_device_name = "ibmq_qasm_simulator"
# ibm_device_name = "default.qubit"
shots_list = [10, 100, 1000, 10000]
num_qubits = 5

qnetti.plot_ibm_network_inference(
    data_dir,
    ibm_device_name,
    shots_list,
    num_qubits,
    prep_node=qnetvo.PrepareNode(wires=[0, 1, 2, 3, 4]),
    title="5-Qubit Zero State",
    cov_mat_match=np.eye(num_qubits),
    mi_char_mat_match=np.zeros((num_qubits, num_qubits)),
    mmi_char_mat_match=np.zeros((num_qubits, num_qubits)),
    # opt_yticks=[
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1,5e-2,1e-2],
    # ],
    avg_data="mi",
)

data_dir = "./data/ibm_inference_W_state_2-qubit_ghz_state_shot_dependence/"
# ibm_device_name = "ibmq_qasm_simulator"
# ibm_device_name = "default.qubit"
ibm_device_name = "ibmq_belem"
shots_list = [10, 100, 1000, 10000]
num_qubits = 5

cov_match = np.array(
    [
        [1, 2 / 3, 2 / 3, 0, 0],
        [2 / 3, 1, 2 / 3, 0, 0],
        [2 / 3, 2 / 3, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
)

w_state_mi = 0.3499775783516452 # {0,1} and {+,-} basis
w_state_vn = 0.9182958340544893 # computational basis
char_match = np.array(
    [
        [w_state_vn, w_state_mi, w_state_mi, 0, 0],
        [w_state_mi, w_state_vn, w_state_mi, 0, 0],
        [w_state_mi, w_state_mi, w_state_vn, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
)


def prep_circ(settings, wires):
    qnetvo.W_state([], wires=wires[0:3])
    qnetvo.ghz_state([], wires=wires[3:5])


w_state_prep_node = qnetvo.PrepareNode(wires=[0, 1, 2, 3, 4], ansatz_fn=prep_circ)

qnetti.plot_ibm_network_inference(
    data_dir,
    ibm_device_name,
    shots_list,
    num_qubits,
    prep_node=w_state_prep_node,
    title="W State and 2-Qubit GHZ State",
    cov_mat_match=cov_match,
    mi_char_mat_match=char_match,
    mmi_char_mat_match=char_match,
    # opt_yticks=[
    #     [5,1,5e-1,1e-1,5e-2,1e-2],
    #     [5,1,5e-1],
    #     [5,1,5e-1,1e-1,5e-2],
    #     [5,1],
    #     [5,1,5e-1,1e-1],
    #     [5,1,5e-1,1e-1],
    # ],
)

qnetti.plot_qubit_inference_heat_map(
    data_dir,
    ibm_device_name,
    title="W State 2-Qubit GHZ State",
    cov_mat_match=cov_match,
    char_mat_match=char_match,
)
