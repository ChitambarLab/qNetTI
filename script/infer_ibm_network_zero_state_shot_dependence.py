import pennylane as qml
from qiskit import IBMQ
import qnetvo
import pennylane_qiskit

from matplotlib import pyplot as plt

import qnetti


# NOTE : uncomment the following if setting up your account for the first time.

# token = "XYZ"   # secret IBM Q API token from above
# IBMQ.save_account(
#     token=token, hub="ibm-q", group="open", project="main", overwrite=True    # open-access account
#     # token=token, hub="ibm-q-ornl", group="ornl", project="chm185"    # DOE/ORNL account
# )

# NOTE : use the next line if you have previously enabled your account using the above lines
provider = IBMQ.load_account()

# NOTE : when running on hardware replace "ibmq_qasm_simulator" with IBM device name
ibm_device_name = "ibmq_qasm_simulator"

shots_list = [10, 100, 1000, 10000]
num_qubits = 5

prep_node = qnetvo.PrepareNode(wires=range(num_qubits))
prep_node_name = "5-qubit_zero_state"


# qnetti.read_json("data/ibm_inference_5-qubit_zero_state_shot_dependence/shots_100/tmp/...")
init_data_json = {}

data_jsons = qnetti.infer_ibm_network_shot_dependence(
    provider,
    prep_node,
    ibm_device_name=ibm_device_name,
    shots_list=shots_list,
    prep_node_name=prep_node_name,
    init_data_json=init_data_json,
)
