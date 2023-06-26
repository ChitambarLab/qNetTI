import pennylane as qml
from qiskit import IBMQ
import qnetvo
import pennylane_qiskit
import sys

from matplotlib import pyplot as plt

import qnetti

"""
Script Positional Command Line Arguments:

1. Required, The first arg should be set as either ``cov``, ``vn``, ``mi``, or ``mmi``
   to collect data for covariance, von neumann entropy, mutual information, or measured mutual information.
2. Optional, default 0, an integer value 0, 1, 2, or 3 that specifies to start the optimization on shots
   10, 100, 1000, or 10000 respectively. This option should only be used if a file name was also provided.
3. Optional, a relative path to an optimization dictionary used to warm start the optimization.

To restart a covariance optimization that failed partway through the optimization using 1000 shots
where note the integer two as the 2nd positional argument specifying steps 1000:

```
python script/infer_ibm_network_ghz_shot_dependence.py cov 2 data/ibm_inference_5-qubit_ghhz_state_shot_dependence/shots_1000/cov/tmp/ibmq_qasm_simulator_<date_time>.json
```
"""

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
# ibm_device_name = "default.qubit"

shots_list = [10, 100, 1000, 10000]
num_qubits = 5

prep_node = qnetvo.PrepareNode(wires=range(num_qubits), ansatz_fn=qnetvo.ghz_state)
prep_node_name = "5-qubit_ghz_state"

kwargs = {}
init_json = {}
if len(sys.argv) > 2:
    shots_list = shots_list[int(sys.argv[2]) :]
    if len(sys.argv) > 3:
        init_json = qnetti.read_json(sys.argv[3])
        kwargs["warm_start_step"] = len(init_json["opt_step_times"])

if sys.argv[1] == "cov":
    kwargs["num_cov_steps"] = 20
    kwargs["cov_init_json"] = init_json
elif sys.argv[1] == "vn":
    kwargs["num_vn_steps"] = 10
    kwargs["vn_init_json"] = init_json
elif sys.argv[1] == "mi":
    kwargs["num_mi_steps"] = 30
    kwargs["mi_init_json"] = init_json
elif sys.argv[1] == "mmi":
    kwargs["num_mmi_steps"] = 30
    kwargs["mmi_init_json"] = init_json

data_jsons = qnetti.infer_ibm_network_shot_dependence(
    provider,
    prep_node,
    ibm_device_name=ibm_device_name,
    shots_list=shots_list,
    prep_node_name=prep_node_name,
    **kwargs,
)
