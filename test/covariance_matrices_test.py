import pytest
import numpy as np
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
        prep_node = qnetvo.PrepareNode(wires=range(num_wires), ansatz_fn=qnetvo.ghz_state)
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node)

        assert np.allclose(cov_mat_fn(meas_settings), cov_mat_match)

    def test_qubit_covariance_matrix_W_state(self):
        prep_node = qnetvo.PrepareNode(wires=[0, 1, 2], ansatz_fn=qnetvo.W_state)
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node)

        assert np.allclose(
            cov_mat_fn(np.zeros(9)), np.array([[8, -4, -4], [-4, 8, -4], [-4, -4, 8]]) / 9
        )

    @pytest.mark.parametrize(
        "angle, cov_match",
        [(np.pi / 2, np.ones((3, 3))), (np.pi / 4, np.ones((3, 3)) / 2)],
    )
    def test_qubit_covariance_matrix_shared_coin_flip_state(self, angle, cov_match):
        prep_node = qnetvo.PrepareNode(
            wires=[0, 1, 2, 3],
            ansatz_fn=lambda settings, wires: qnetvo.shared_coin_flip_state([angle], wires),
        )
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node, meas_wires=[0, 1, 2])

        assert np.allclose(cov_mat_fn(np.zeros(9)), cov_match)

    @pytest.mark.parametrize(
        "num_wires, meas_settings, cov_mat_match",
        [
            (2, [0, 0, 0, 0, 0, 0], [[1, 1], [1, 1]]),
            (2, [0, np.pi / 4, 0, 0, 0, 0], [[1, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1]]),
            (2, [0, np.pi / 2, 0, 0, 0, 0], [[1, 0], [0, 1]]),
            (2, [0, np.pi / 2, 0, 0, np.pi / 2, 0], [[1, 1], [1, 1]]),
            (2, [0, 0, 0, 0, np.pi, 0], [[1, -1], [-1, 1]]),
        ],
    )
    def test_qubit_covariance_matrix_finite_shots(self, num_wires, meas_settings, cov_mat_match):
        np.random.seed(65)
        prep_node = qnetvo.PrepareNode(wires=range(num_wires), ansatz_fn=qnetvo.ghz_state)
        cov_mat_fn = qnetti.qubit_covariance_matrix_fn(prep_node, dev_kwargs={"shots": 20000})

        assert np.allclose(cov_mat_fn(meas_settings), cov_mat_match, atol=1e-2)


class TestQubitCovarianceCostFn:
    @pytest.mark.parametrize(
        "num_wires, meas_settings, meas_wires, cost_match",
        [
            (2, [0, 0, 0, 0, 0, 0], None, -4),
            (2, [0, np.pi / 4, 0, 0, 0, 0], None, -3),
            (2, [0, np.pi / 2, 0, 0, 0, 0], None, -2),
            (2, [0, np.pi / 2, 0, 0, np.pi / 2, 0], None, -4),
            (2, [0, 0, 0, 0, np.pi, 0], None, -4),
            (3, [0, 0, 0, 0, 0, 0, 0, 0, 0], None, -9),
            (3, [0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0], None, -7),
            (3, [0, np.pi / 2, 0, 0, 0, 0, 0, np.pi, 0], None, -5),
            (3, [0, 0, 0, 0, 0, 0], [0, 1], -4),
        ],
    )
    def test_qubit_covariance_cost_fn_ghz_state(
        self, num_wires, meas_settings, meas_wires, cost_match
    ):
        prep_node = qnetvo.PrepareNode(wires=range(num_wires), ansatz_fn=qnetvo.ghz_state)
        cov_cost = qnetti.qubit_covariance_cost_fn(prep_node, meas_wires=meas_wires)

        assert np.allclose(cov_cost(meas_settings), cost_match)

    @pytest.mark.parametrize(
        "num_wires, meas_settings, meas_wires, cost_match",
        [
            (2, [0, 0, 0, 0, 0, 0], None, -4),
            (2, [0, np.pi / 4, 0, 0, 0, 0], None, -3),
            (2, [0, np.pi / 2, 0, 0, 0, 0], None, -2),
            (2, [0, np.pi / 2, 0, 0, np.pi / 2, 0], None, -4),
            (2, [0, 0, 0, 0, np.pi, 0], None, -4),
            (3, [0, 0, 0, 0, 0, 0, 0, 0, 0], None, -9),
            (3, [0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0], None, -7),
            (3, [0, np.pi / 2, 0, 0, 0, 0, 0, np.pi, 0], None, -5),
            (3, [0, 0, 0, 0, 0, 0], [0, 1], -4),
        ],
    )
    def test_qubit_covariance_cost_fn_finite_shot(
        self, num_wires, meas_settings, meas_wires, cost_match
    ):
        prep_node = qnetvo.PrepareNode(wires=range(num_wires), ansatz_fn=qnetvo.ghz_state)
        cov_cost = qnetti.qubit_covariance_cost_fn(
            prep_node, meas_wires=meas_wires, dev_kwargs={"shots": 20000}
        )

        assert np.allclose(cov_cost(meas_settings), cost_match, atol=1e-2)
