import pennylane as qml
from pennylane import numpy as qnp
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
import qnetvo
import pennylane_qiskit

from matplotlib import pyplot as plt

import qnetti

"""
This script infers the topology of a 5-qubit ghz state via the measured mutal info characteristic matrix.
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
filepath = qnetti.mkdir("./data/", "ibm_expt_ghz_state_mmi_char_shot_dependence/")

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

for shots in shots_list:
    shots_filepath = qnetti.mkdir(filepath, "shots_" + str(shots) + "/")
    dev_kwargs["shots"] = shots

    mi_opt_kwargs = {
        "num_steps": 20,
        "step_size": 0.1,
        "filepath": shots_filepath,
    }

    vn_opt_kwargs = {
        "num_steps": 10,
        "step_size": 0.1,
        "filepath": shots_filepath,
    }

    datetime_str = qnetti.datetime_now_string()
    char_mat, mi_opt_dict, vn_opt_dict = qnetti.optimize_characteristic_matrix(
        qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state),
        dev_kwargs=dev_kwargs,
        use_measured_mutual_info=True,
        meas_wires=wires,
        qnode_kwargs=qnode_kwargs,
        mi_opt_kwargs=mi_opt_kwargs,
        vn_opt_kwargs=vn_opt_kwargs,
    )

    qnetti.write_json(
        {
            "char_mat": char_mat.tolist(),
            "mmi_opt_dict": mi_opt_dict,
            "vn_opt_dict": vn_opt_dict,
            "ibm_device": ibm_device_name,
            "num_qubits": num_qubits,
        },
        shots_filepath + ibm_device_name + "_char_mat_" + datetime_str,
    )

    print("Characteristic Matrix : ", char_mat)

    ax1.plot(
        range(len(vn_opt_dict["cost_vals"])),
        vn_opt_dict["cost_vals"],
        label="shots " + str(shots),
    )
    ax2.plot(
        range(len(mi_opt_dict["cost_vals"])),
        mi_opt_dict["cost_vals"],
        label="shots " + str(shots),
    )

fig.suptitle("Mutual Info Inferece of " + str(num_qubits) + "-Qubit GHZ state")
plt.legend()

ax1.set_title("Von Neumann Entropy Optimization")
ax1.set_xlabel("Optimization Step")
ax1.set_ylabel(r"Cost")

ax2.set_title("Measured Mutual Info Optimization")
ax2.set_xlabel("Optimization Step")
ax2.set_ylabel(r"Cost")

plt.show()
