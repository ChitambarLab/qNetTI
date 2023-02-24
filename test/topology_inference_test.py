import pytest
import qnetvo
import pennylane as qml
import pennylane.numpy as np
import qnetti


@pytest.mark.parametrize(
    "matrix, output_match",
    [
        (np.zeros((3, 3)), [[0, 1, 2]]),
        (np.ones((3, 3)), [[0, 1, 2]]),
        (np.kron(np.eye(2), np.ones((2, 2))), [[0, 1], [2, 3]]),
        ([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], [[0, 2], [1, 3]]),
        ([[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1], [2], [3]]),
        ([[1, 0.001, 0.05], [-0.004, 1, 0.001], [0.06, 0.001, 1]], [[0, 2], [1]]),
    ],
)
def test_characteristic_matrix_decoder(matrix, output_match):
    assert qnetti.characteristic_matrix_decoder(matrix, tol=1e-2) == output_match
