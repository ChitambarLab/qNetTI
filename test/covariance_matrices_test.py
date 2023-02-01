import pytest
import numpy as np
import pennylane as qml
import qnetvo

import qnetti


class TestQubitCovarianceMatrixFn:
    @pytest.mark.parametrize(
        "num_wires, meas_settings, cov_mat_match",
        [
            (2, [0, 0, 0, 0, 0, 0], [[1, 1], [1, 1]]),
            (2, [0, np.pi / 4, 0, 0, 0, 0], [[1, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1]]),
            (2, [0, np.pi / 2, 0, 0, 0, 0], [[1, 0], [0, 1]]),
            (2, [0, np.pi / 2, 0, 0, np.pi / 2, 0], [[1, 1], [1, 1]]),
            (2, [0, 0, 0, 0, np.pi, 0], [[1, -1], [-1, 1]]),
            (3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            (
                3,
                [0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0],
                [
                    [1, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                    [1 / np.sqrt(2), 1, 1],
                    [1 / np.sqrt(2), 1, 1],
                ],
            ),
            (3, [0, np.pi / 2, 0, 0, 0, 0, 0, np.pi, 0], [[1, 0, 0], [0, 1, -1], [0, -1, 1]]),
        ],
    )
    def test_qubit_covariance_matrix_ghz_state(self, num_wires, meas_settings, cov_mat_match):
        prep_node = qnetvo.PrepareNode(
            num_in=1, wires=range(num_wires), ansatz_fn=qnetvo.ghz_state, num_settings=0
        )
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node)

        assert np.allclose(cov_mat_fn(meas_settings), cov_mat_match)

    def test_qubit_covariance_matrix_W_state(self):
        def W_state(settings, wires):
            phi = 2 * np.arccos(1 / np.sqrt(3))
            qml.RY(phi, wires=wires[0])
            qml.CRot(-2 * np.pi, np.pi / 2, 2 * np.pi, wires=wires[0:2])

            qml.CNOT(wires=wires[1:3])
            qml.CNOT(wires=wires[0:2])
            qml.PauliX(wires=wires[0])

        prep_node = qnetvo.PrepareNode(num_in=1, wires=[0, 1, 2], ansatz_fn=W_state, num_settings=0)
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node)

        assert np.allclose(
            cov_mat_fn(np.zeros(9)), np.array([[8, -4, -4], [-4, 8, -4], [-4, -4, 8]]) / 9
        )


class TestQubitCovarianceCostFn:
    @pytest.mark.parametrize(
        "num_wires, meas_settings, cost_match",
        [
            (2, [0, 0, 0, 0, 0, 0], -4),
            (2, [0, np.pi / 4, 0, 0, 0, 0], -3),
            (2, [0, np.pi / 2, 0, 0, 0, 0], -2),
            (2, [0, np.pi / 2, 0, 0, np.pi / 2, 0], -4),
            (2, [0, 0, 0, 0, np.pi, 0], -4),
            (3, [0, 0, 0, 0, 0, 0, 0, 0, 0], -9),
            (3, [0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0], -7),
            (3, [0, np.pi / 2, 0, 0, 0, 0, 0, np.pi, 0], -5),
        ],
    )
    def test_qubit_covariance_cost_fn_ghz_state(self, num_wires, meas_settings, cost_match):
        prep_node = qnetvo.PrepareNode(
            num_in=1, wires=range(num_wires), ansatz_fn=qnetvo.ghz_state, num_settings=0
        )
        cov_cost = qnetti.qubit_covariance_cost_fn(prep_node)

        assert np.allclose(cov_cost(*meas_settings), cost_match)
