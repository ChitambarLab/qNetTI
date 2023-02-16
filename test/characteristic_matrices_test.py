import pytest
import numpy as np
from pennylane import numpy as qnp
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
        "wires, mi_settings, match",
        [
            ([0, 1], [0] * 6, np.ones((2, 2))),
            ([0, 1], [np.pi / 2] + [0] * 5, np.eye(2)),
            ([0, 1, 2], [np.pi / 2] + [0] * 8, [[1, 0, 0], [0, 1, 1], [0, 1, 1]]),
            ([0, 1, 2], [np.pi / 2, 0, 0] * 3, np.eye(3)),
        ],
    )
    def test_qubit_characteristic_matrix_fn_ghz_states(self, wires, mi_settings, match):
        prep_node = qnetvo.PrepareNode(wires=wires, ansatz_fn=qnetvo.ghz_state)
        char_mat = qnetti.qubit_characteristic_matrix_fn(prep_node)
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


class TestOptimizeVnEntropy:
    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-6), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_vn_entropy_matrix_two_qubit_ghz_state(self, dev_kwargs, atol, qnode_kwargs):
        prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
        qnp.random.seed(55)

        opt_dict = qnetti.optimize_vn_entropy(
            prep_node,
            dev_kwargs=dev_kwargs,
            step_size=0.1,
            num_steps=5,
            qnode_kwargs=qnode_kwargs,
        )

        assert np.isclose(opt_dict["min_cost"], 2, atol=atol)
        assert len(opt_dict["cost_vals"]) == 6

    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_vn_entropy_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
        qnp.random.seed(32)
        num_steps = 25
        opt_dict = qnetti.optimize_vn_entropy(
            qnetvo.PrepareNode(wires=[0, 1]),
            dev_kwargs=dev_kwargs,
            step_size=0.2,
            num_steps=num_steps,
            qnode_kwargs=qnode_kwargs,
        )

        assert np.isclose(opt_dict["min_cost"], 0, atol=atol)
        assert len(opt_dict["cost_vals"]) == num_steps + 1


class TestOptimizeMutualInfo:
    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_mutual_info_matrix_two_qubit_ghz_state(self, dev_kwargs, atol, qnode_kwargs):
        prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
        qnp.random.seed(55)

        opt_dict = qnetti.optimize_mutual_info(
            prep_node,
            dev_kwargs=dev_kwargs,
            step_size=0.1,
            num_steps=15,
            qnode_kwargs=qnode_kwargs,
        )

        assert np.isclose(opt_dict["min_cost"], -1, atol=atol)
        assert len(opt_dict["cost_vals"]) == 16

    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_mutual_info_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
        qnp.random.seed(32)
        num_steps = 12
        opt_dict = qnetti.optimize_mutual_info(
            qnetvo.PrepareNode(wires=[0, 1]),
            dev_kwargs=dev_kwargs,
            step_size=0.1,
            num_steps=num_steps,
            qnode_kwargs=qnode_kwargs,
        )

        assert np.isclose(opt_dict["min_cost"], 0, atol=atol)
        assert len(opt_dict["cost_vals"]) == num_steps + 1


class TestOptimizeCharacteristicMatrix:
    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_characteristic_matrix_two_qubit_ghz_state(
        self, dev_kwargs, atol, qnode_kwargs
    ):
        prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
        qnp.random.seed(55)

        char_mat, mi_opt_dict, vn_opt_dict = qnetti.optimize_characteristic_matrix(
            prep_node,
            cost_kwargs={
                "dev_kwargs": dev_kwargs,
                "qnode_kwargs": qnode_kwargs,
            },
            mi_opt_kwargs={
                "step_size": 0.1,
                "num_steps": 15,
            },
            vn_opt_kwargs={
                "step_size": 0.1,
                "num_steps": 12,
            },
        )

        assert np.allclose(char_mat, [[1, 1], [1, 1]], atol=atol)
        assert len(mi_opt_dict["cost_vals"]) == 16
        assert len(vn_opt_dict["cost_vals"]) == 13

    @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_characteristic_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
        qnp.random.seed(32)
        char_mat, mi_opt_dict, vn_opt_dict = qnetti.optimize_characteristic_matrix(
            qnetvo.PrepareNode(wires=[0, 1]),
            cost_kwargs={
                "dev_kwargs": dev_kwargs,
                "qnode_kwargs": qnode_kwargs,
            },
            mi_opt_kwargs={
                "step_size": 0.1,
                "num_steps": 12,
            },
            vn_opt_kwargs={
                "step_size": 0.2,
                "num_steps": 25,
            },
        )

        assert np.allclose(char_mat, np.zeros((2, 2)), atol=atol)
        assert len(mi_opt_dict["cost_vals"]) == 13
        assert len(vn_opt_dict["cost_vals"]) == 26
