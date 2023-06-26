import qnetti
import qnetvo

import pennylane as qml
from pennylane import numpy as qnp
import time

prep_node = qnetvo.PrepareNode(
    wires=range(12),
    ansatz_fn=qnetvo.graph_state_fn([[0, 1], [1, 2], [6, 7], [4, 5], [4, 9], [4, 11], [11, 10]]),
)

start = time.time()
char_mat = qnetti.optimize_characteristic_matrix(
    prep_node,
)
print("char mat evaluated in : ", time.time() - start)

char_mat

start = time.time()
cov_mat = qnetti.optimize_covariance_matrix(
    prep_node,
)
print("char mat evaluated in : ", time.time() - start)


"""
Baseline of current performance
"""

prep_node = qnetvo.PrepareNode(
    wires=range(8),
    ansatz_fn=qnetvo.ghz_state,
)

start = time.time()
char_mat = qnetti.qubit_characteristic_matrix(
    prep_node,
    step_size=0.1,
    num_steps=50,
)

print("char mat evaluated in : ", time.time() - start)

start = time.time()
cov_mat = qnetti.optimize_covariance_matrix(prep_node)

print("cov_mat evaluated in : ", time.time() - start)

"""
testing removal of qnetvo.gradient_descent on performance
"""

start = time.time()

opt_mi_settings = optimize_mutual_info_settings(prep_node, 8)

print("mutual info optimization time ", time.time() - start)

start = time.time()

opt_vn_settings = optimize_vn_entropy_settings(prep_node, 8)

print("vn entropy optimization time ", time.time() - start)

char_mat = qnetti.evaluate_qubit_char_mat(prep_node)(opt_vn_settings, opt_mi_settings)

"""
Evaluating char mat with new methods
"""


"""
More toe to toe comparison
"""

prep_node = qnetvo.PrepareNode(
    wires=range(12),
    ansatz_fn=qnetvo.graph_state_fn([[0, 1], [1, 2], [6, 7], [4, 5], [4, 9], [4, 11], [11, 10]]),
)

start = time.time()
char_mat = qnetti.optimize_characteristic_matrix(
    prep_node,
)
print("char mat evaluated in : ", time.time() - start)
# print(char_mat)

start = time.time()
cov_mat = covariance_matrix_inference(prep_node, 12)
print("cov_mat evaluated in : ", time.time() - start)
# print(cov_mat)

prep_node = qnetvo.PrepareNode(
    wires=range(12),
    ansatz_fn=qnetvo.ghz_state,
)

start = time.time()
char_mat = char_matrix_inference(
    prep_node,
    12,
    step_size=0.1,
    num_steps=10,
)
print("char mat evaluated in : ", time.time() - start)
print(char_mat)

start = time.time()
cov_mat = covariance_matrix_inference(prep_node, 12)
print("cov_mat evaluated in : ", time.time() - start)
print(cov_mat)
