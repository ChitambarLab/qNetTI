import pytest
import numpy as np
import qnetvo

import qnetti


@pytest.mark.parametrize(
    "probs_list, match",
    [
        ([0, 1], [0]),
        ([0.5, 0.5], [1]),
        ([0, 0, 0, 1], [0, 0]),
        ([0.5, 0.5, 0, 0], [0, 1]),
        ([1 / 8] * 8, [1, 1, 1]),
    ],
)
def test_qubit_shannon_entropies(probs_list, match):
    probs_vec = np.array(probs_list)
    shannon_entropies = qnetti.qubit_shannon_entropies(probs_vec)
    assert len(shannon_entropies) == len(match)
    assert np.allclose(shannon_entropies, match)


@pytest.mark.parametrize(
    "probs_list, match",
    [
        ([0, 1], []),
        ([1, 0, 0, 0], [0]),
        ([0.5, 0, 0, 0.5], [1]),
        ([0.25] * 4, [0]),
        ([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0]),
        ([1 / 16] * 16, [0, 0, 0, 0, 0, 0]),
        ([1 / 4, 1 / 4, 0, 0, 0, 0, 1 / 4, 1 / 4], [1, 0, 0]),
    ],
)
def test_qubit_mutual_infos(probs_list, match):
    probs_vec = np.array(probs_list)
    mutual_infos = qnetti.qubit_mutual_infos(probs_vec)

    assert len(mutual_infos) == len(match)
    assert np.allclose(mutual_infos, match)


class TestShannonEntropyCostFn:
    @pytest.mark.parametrize(
        "wires, settings",
        [
            ([0, 1], [0, 0, 0, 0, 0, 0]),
            ([0, 1], [np.pi / 2, 0, 0, 0, 0, 0]),
            ([0, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ([0, 1, 2], [0] * 9),
            ([0, 1, 2], [0.1, 0.2, 0.3] * 3),
            ([0, 1, 2, 3], [0] * 12),
        ],
    )
    def test_shannon_entropy_cost_fn_ghz_states(self, wires, settings):
        prep_node = qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state)
        shannon_entropy_cost = qnetti.shannon_entropy_cost_fn(prep_node)
        assert np.isclose(shannon_entropy_cost(settings), len(wires))

    @pytest.mark.parametrize(
        "wires, settings",
        [
            ([0, 1, 2], [0, 0, 0, 0, 0, 0]),
            ([0, 1, 2], [np.pi / 2, 0, 0, 0, 0, 0]),
            ([0, 1, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ([0, 1, 2, 3], [0] * 9),
            ([0, 1, 2, 3], [0.1, 0.2, 0.3] * 3),
            ([0, 1, 2, 3, 4], [0] * 12),
        ],
    )
    def test_shannon_entropy_cost_fn_shared_randomness(self, wires, settings):
        prep_node = qnetvo.PrepareNode(
            wires=wires,
            ansatz_fn=lambda settings, wires: qnetvo.shared_coin_flip_state(
                [np.pi / 2], wires=wires
            ),
        )
        shannon_entropy_cost = qnetti.shannon_entropy_cost_fn(prep_node, meas_wires=wires[0:-1])
        assert np.isclose(shannon_entropy_cost(settings), len(wires) - 1)

    @pytest.mark.parametrize(
        "wires, settings, match",
        [
            ([0], [0, 0, 0], 0),
            ([0], [np.pi / 2, 0, 0], 1),
            ([0, 1, 2], [np.pi / 2, 0, 0] + [0] * 6, 1),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 3, 3),
        ],
    )
    def test_shannon_entropy_cost_fn_separable_states(self, wires, settings, match):
        prep_node = qnetvo.PrepareNode(wires=wires)
        shannon_entropy_cost = qnetti.shannon_entropy_cost_fn(prep_node)
        assert np.isclose(shannon_entropy_cost(settings), match)


class TestMeasuredMutualInfoCostFn:
    @pytest.mark.parametrize(
        "wires, settings, match",
        [
            ([0, 1], [0] * 6, [1]),
            ([0, 1, 2], [0] * 18, [1, 1, 1]),
            ([0, 1, 2, 3], [0] * 36, [1, 1, 1, 1, 1, 1]),
            ([0, 1], [0, 0, 0, np.pi / 2, 0, 0], [0]),
            (
                [0, 1, 2],
                [0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0, np.pi / 2, 0, 0],
                [0, 1, 0],
            ),
        ],
    )
    def test_measured_mutual_infos_and_cost_fn_ghz_state(self, wires, settings, match):
        prep_node = qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state)
        measured_mutual_infos = qnetti.qubit_measured_mutual_infos_fn(prep_node)

        assert np.allclose(measured_mutual_infos(settings), match)

        measured_mutual_infos_cost = qnetti.measured_mutual_info_cost_fn(prep_node)

        assert np.isclose(measured_mutual_infos_cost(settings), -sum(match))


class TestMutualInfoCostFn:
    @pytest.mark.parametrize(
        "wires, settings, match",
        [
            ([0, 1, 2], [0, 0, 0, 0, 0, 0], -1),
            ([0, 1, 2], [np.pi / 2, 0, 0, 0, 0, 0], 0),
            ([0, 1, 2], [np.pi / 2, 0, 0, np.pi / 2, 0, 0], 0),
            ([0, 1, 2, 3], [0] * 9, -3),
            ([0, 1, 2, 3], [0, np.pi / 2, 0] * 3, 0),
            ([0, 1, 2, 3], [0, np.pi / 2, 0] + [0] * 6, -1),
            ([0, 1, 2, 3], [0, np.pi / 2, 0] * 2 + [0] * 3, 0),
            ([0, 1, 2, 3, 4], [0] * 12, -6),
        ],
    )
    def test_mutual_info_cost_fn_shared_randomness_states(self, wires, settings, match):
        prep_node = qnetvo.PrepareNode(
            wires=wires,
            ansatz_fn=lambda settings, wires: qnetvo.shared_coin_flip_state(
                [np.pi / 2], wires=wires
            ),
        )
        mutual_info_cost = qnetti.mutual_info_cost_fn(prep_node, meas_wires=wires[0:-1])
        assert np.isclose(mutual_info_cost(settings), match)

    @pytest.mark.parametrize(
        "wires, settings, match",
        [
            ([0, 1], [0, 0, 0, 0, 0, 0], -1),
            ([0, 1], [np.pi / 2, 0, 0, 0, 0, 0], 0),
            ([0, 1], [np.pi / 2, 0, 0, np.pi / 2, 0, 0], -1),
            ([0, 1, 2], [0] * 9, -3),
            ([0, 1, 2], [0, np.pi / 2, 0] * 3, 0),
            ([0, 1, 2], [0, np.pi / 2, 0] + [0] * 6, -1),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 2 + [0] * 3, 0),
            ([0, 1, 2, 3], [0] * 12, -6),
        ],
    )
    def test_mutual_info_cost_fn_ghz_states(self, wires, settings, match):
        prep_node = qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state)
        mutual_info_cost = qnetti.mutual_info_cost_fn(prep_node)
        assert np.isclose(mutual_info_cost(settings), match)

    @pytest.mark.parametrize(
        "wires, settings",
        [
            ([0, 1], [0] * 6),
            ([0, 1], [np.pi / 2, 0, 0, 0, 0, 0]),
            ([0, 1, 2], [0] * 9),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 3),
        ],
    )
    def test_shannon_entropy_cost_fn_separable_states(self, wires, settings):
        prep_node = qnetvo.PrepareNode(wires=wires)
        shannon_entropy_cost = qnetti.mutual_info_cost_fn(prep_node)
        assert np.isclose(shannon_entropy_cost(settings), 0)


class TestQubitCharacteristicFn:
    @pytest.mark.parametrize(
        "wires, mi_settings, use_measured_mutual_info, match",
        [
            ([0, 1], [0] * 6, False, np.ones((2, 2))),
            ([0, 1], [np.pi / 2] + [0] * 5, False, np.eye(2)),
            ([0, 1, 2], [np.pi / 2] + [0] * 8, False, [[1, 0, 0], [0, 1, 1], [0, 1, 1]]),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 3, False, np.eye(3)),
            ([0, 1], [0] * 6, True, np.ones((2, 2))),
            (
                [0, 1, 2],
                [np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0],
                True,
                [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            ),
        ],
    )
    def test_qubit_characteristic_matrix_fn_ghz_states(
        self, wires, mi_settings, use_measured_mutual_info, match
    ):
        prep_node = qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state)
        char_mat = qnetti.qubit_characteristic_matrix_fn(
            prep_node, use_measured_mutual_info=use_measured_mutual_info
        )
        vn_settings = [0] * 3 * len(wires)
        assert np.allclose(char_mat(vn_settings, mi_settings), match)

    @pytest.mark.parametrize(
        "wires, vn_settings, match",
        [
            ([0, 1], [0] * 6, np.zeros((2, 2))),
            ([0, 1], [np.pi / 2] + [0] * 5, [[1, 0], [0, 0]]),
            ([0, 1, 2], [np.pi / 2] + [0] * 8, [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 3, np.eye(3)),
        ],
    )
    def test_qubit_characteristic_matrix_fn_separable_states(self, wires, vn_settings, match):
        prep_node = qnetvo.PrepareNode(wires=wires)
        char_mat = qnetti.qubit_characteristic_matrix_fn(prep_node)
        mi_settings = [0] * 3 * len(wires)
        assert np.allclose(char_mat(vn_settings, mi_settings), match)
