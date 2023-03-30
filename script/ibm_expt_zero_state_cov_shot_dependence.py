import pennylane as qml
from pennylane import numpy as qnp
from qiskit import IBMQ
import qnetvo
import pennylane_qiskit

from matplotlib import pyplot as plt

import qnetti

"""
This script infers the topology of a 5-qubit zero state.
The script must be edited before using:
    
    1. Add the IBM Q API token, and configure the IBMQ provider
    2. Set the ibm_device_name to the reserved IBM device.

Note that data saved upon competion and a plot is generated.
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

dev_kwargs = {
    "name": "qiskit.ibmq",
    "backend": ibm_device_name,
    "provider": provider,
}

shots_list = [10, 100, 1000, 10000]
num_qubits = 5


wires = range(num_qubits)
qnode_kwargs = {"diff_method": "parameter-shift"}
filepath = qnetti.mkdir("./data/", "ibm_expt_zero_state_cov_shot_dependence/")

for shots in shots_list:
    shots_filepath = qnetti.mkdir(filepath, "shots_" + str(shots) + "/")
    dev_kwargs["shots"] = shots

    datetime_str = qnetti.datetime_now_string()
    cov_mat, cov_opt_dict = qnetti.optimize_covariance_matrix(
        qnetvo.PrepareNode(wires=wires),
        dev_kwargs=dev_kwargs,
        meas_wires=wires,
        qnode_kwargs=qnode_kwargs,
        num_steps=20,
        step_size=0.1,
        filepath=shots_filepath,
    )

    qnetti.write_json(
        {
            "cov_mat": cov_mat.tolist(),
            "opt_dict": cov_opt_dict,
            "ibm_device": ibm_device_name,
            "num_qubits": num_qubits,
        },
        shots_filepath + ibm_device_name + "_cov_mat_" + datetime_str,
    )

    print("Covariance Matrix : ", cov_mat)

    plt.plot(
        range(len(cov_opt_dict["cost_vals"])),
        cov_opt_dict["cost_vals"],
        label="shots " + str(shots),
    )

plt.title("Covariance Inferece of " + str(num_qubits) + "-Qubit Zero state")
plt.legend()
plt.xlabel("Optimization Step")
plt.ylabel(r"Covariance Cost $Tr[M^T M]$")
plt.show()
