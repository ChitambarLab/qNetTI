import pennylane as qml
from pennylane import numpy as qnp

import qnetvo
import qnetti


"""
Code to optimize covaariance matrix without finite shots
"""

prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
analytic_dev_kwargs = {
    "name": "default.qubit",
}
cov_cost = qnetti.qubit_covariance_cost_fn(prep_node, dev_kwargs=analytic_dev_kwargs)

opt = qml.GradientDescentOptimizer(stepsize=0.1)

settings_list = []
cost_vals = []
settings = qnp.random.rand(6, requires_grad=True)

for i in range(10):
    settings, cost_val = opt.step_and_cost(cov_cost, settings)
    cost_vals += [cost_val]
    settings_list += [settings]

print(cost_vals)

# opt_dict = qnetvo.gradient_descent(
#     cov_cost,
#     qnp.random.rand(6, requires_grad=True),
#     step_size=0.1,
#     num_steps=10,
#     sample_width=1,
#     verbose=False,
# )

print("opt result : ", opt_dict["opt_score"])


"""
Code to reproduce finite shot bug
"""

prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
# finite_shot_dev_kwargs = {
#     "name": "default.qubit",
#     "shots": 1000,
# }
cov_cost = qnetti.qubit_covariance_cost_fn(
    prep_node, shots=1000, qnode_kwargs={"diff_method": "parameter-shift"}
)

for i in range(10):
    settings, cost_val = opt.step_and_cost(cov_cost, settings)
    cost_vals += [cost_val]
    settings_list += [settings]

print(cost_vals)

# opt_dict = qnetvo.gradient_descent(
#     cov_cost,
#     qnp.random.rand(6, requires_grad=True),
#     step_size=0.1,
#     num_steps=10,
#     sample_width=1,
#     verbose=False,
# )

print("opt result : ", opt_dict["opt_score"])

"""
Implementation without qnetvo
"""

finite_shot_dev_kwargs = {
    "name": "default.qubit",
    # "shots": 1000,
    "wires": [0, 1],
}
finite_shot_dev = qml.device(**finite_shot_dev_kwargs)

import pennylane as qml
from pennylane import numpy as qnp

dev_kwargs = {
    "name": "default.qubit",
    "wires": [0, 1],
    "shots": 1000,
}

dev = qml.device(**dev_kwargs)
# dev = qml.device("default.qubit", [0,1], shots=1000)


@qml.qnode(dev, diff_method="parameter-shift")
def circ(settings):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])

    qml.ArbitraryUnitary(settings[0:3], wires=[0])
    qml.ArbitraryUnitary(settings[3:6], wires=[1])

    return qml.probs(wires=[0, 1])


init_settings = qnp.random.rand(6, requires_grad=True)
circ(init_settings)


def cost(settings):
    circ_probs = circ(settings)
    cov_mat = qml.math.cov_matrix(circ_probs, obs=[qml.PauliZ(0), qml.PauliZ(1)])

    return -qnp.trace(cov_mat.T @ cov_mat)


cost(init_settings)


opt = qml.GradientDescentOptimizer(stepsize=0.1)

settings_list = []
cost_vals = []
settings = init_settings

for i in range(10):
    settings, cost_val = opt.step_and_cost(cost, settings)
    cost_vals += [cost_val]
    settings_list += [settings]

print(cost_vals)
