import pytest
import pennylane as qml
from pennylane import numpy as qnp
import qnetvo

import qnetti


@pytest.fixture
def frustrated_mixed_state_prep_node():
    """
    This state does not allow simultaneous correlation between all qubit pairs.
    The bipartite measured mutual information admits more correlation structure than
    standard covariance and mutual infomration.
    """

    rho = [
        [0.1875, 0.0, 0.0, 0.0, 0.0, -0.0625, 0.0625, 0.0],
        [0.0, 0.0625, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0625],
        [0.0, 0.0, 0.0625, 0.0, 0.0625, 0.0, 0.0, -0.0625],
        [0.0, 0.0, 0.0, 0.1875, 0.0, 0.0625, 0.0625, 0.0],
        [0.0, 0.0625, 0.0625, 0.0, 0.1875, 0.0, 0.0, 0.0],
        [-0.0625, 0.0, 0.0, 0.0625, 0.0, 0.0625, 0.0, 0.0],
        [0.0625, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0625, 0.0],
        [0.0, 0.0625, -0.0625, 0.0, 0.0, 0.0, 0.0, 0.1875],
    ]

    X = qnp.array([[0, 1], [1, 0]])
    Y = qnp.array([[0, -1j], [1j, 0]])
    Z = qnp.array([[1, 0], [0, -1]])

    rho_match = (
        qnp.eye(8) / 8
        + qnp.kron(X, qnp.kron(X, qnp.eye(2))) / 16
        + qnp.kron(Y, qnp.kron(qnp.eye(2), Y)) / 16
        + qnp.kron(qnp.eye(2), qnp.kron(Z, Z)) / 16
    )

    assert qnp.allclose(rho, rho_match)
    assert qnp.isclose(qnp.trace(rho), 1)
    assert all(map(lambda x: x >= 0 or qnp.isclose(x, 0), qnp.linalg.eigvals(rho)))

    def mixed_circ(settings, wires):
        qml.QubitDensityMatrix(rho, wires=wires)

    return qnetvo.PrepareNode(wires=[0, 1, 2], ansatz_fn=mixed_circ)


class TestOptimizeCovarianceMatrix:
    @pytest.mark.parametrize(
        "qnode_kwargs",
        [{}, {"diff_method": "parameter-shift"}],
    )
    def test_optimize_covariance_matrix_two_qubit_ghz_state(self, qnode_kwargs):
        prep_node = qnetvo.PrepareNode(wires=range(2), ansatz_fn=qnetvo.ghz_state)
        qnp.random.seed(55)

        cov_mat, opt_dict = qnetti.optimize_covariance_matrix(
            prep_node,
            step_size=0.1,
            num_steps=10,
            dev_kwargs={"shots": 1000},
            qnode_kwargs=qnode_kwargs,
        )

        assert qnp.isclose(opt_dict["min_cost"], -4, atol=1e-2)
        assert qnp.allclose(cov_mat, qnp.ones((2, 2)), atol=1e-3)

    def test_optimize_covariance_matrix_frustrated_mixed_state(
        self, frustrated_mixed_state_prep_node
    ):
        qnp.random.seed(11)
        cov_mat, opt_dict = qnetti.optimize_covariance_matrix(
            frustrated_mixed_state_prep_node,
            step_size=0.5,
            num_steps=30,
            dev_kwargs={"name": "default.mixed"},
        )

        assert qnp.allclose(cov_mat, [[1, 0, -0.218], [0, 1, -0.45], [-0.218, -0.45, 1]], atol=2e-3)


# class TestOptimizeVnEntropy:
#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-6), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_vn_entropy_matrix_two_qubit_ghz_state(self, dev_kwargs, atol, qnode_kwargs):
#         prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
#         qnp.random.seed(55)

#         opt_dict = qnetti.optimize_vn_entropy(
#             prep_node,
#             dev_kwargs=dev_kwargs,
#             step_size=0.1,
#             num_steps=5,
#             qnode_kwargs=qnode_kwargs,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], 2, atol=atol)
#         assert len(opt_dict["cost_vals"]) == 6

#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_vn_entropy_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
#         qnp.random.seed(32)
#         num_steps = 25
#         opt_dict = qnetti.optimize_vn_entropy(
#             qnetvo.PrepareNode(wires=[0, 1]),
#             dev_kwargs=dev_kwargs,
#             step_size=0.2,
#             num_steps=num_steps,
#             qnode_kwargs=qnode_kwargs,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], 0, atol=atol)
#         assert len(opt_dict["cost_vals"]) == num_steps + 1


# class TestOptimizeMutualInfo:
#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_mutual_info_matrix_two_qubit_ghz_state(self, dev_kwargs, atol, qnode_kwargs):
#         prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
#         qnp.random.seed(55)

#         opt_dict = qnetti.optimize_mutual_info(
#             prep_node,
#             dev_kwargs=dev_kwargs,
#             step_size=0.1,
#             num_steps=15,
#             qnode_kwargs=qnode_kwargs,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], -1, atol=atol)
#         assert len(opt_dict["cost_vals"]) == 16

#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_mutual_info_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
#         qnp.random.seed(32)
#         num_steps = 12
#         opt_dict = qnetti.optimize_mutual_info(
#             qnetvo.PrepareNode(wires=[0, 1]),
#             dev_kwargs=dev_kwargs,
#             step_size=0.1,
#             num_steps=num_steps,
#             qnode_kwargs=qnode_kwargs,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], 0, atol=atol)
#         assert len(opt_dict["cost_vals"]) == num_steps + 1


# class TestOptimizeMeasuredMutualInfo:
#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_measured_mutual_info_two_qubit_ghz_state(
#         self, dev_kwargs, atol, qnode_kwargs
#     ):
#         qnp.random.seed(35)

#         opt_dict = qnetti.optimize_measured_mutual_info(
#             qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state),
#             dev_kwargs=dev_kwargs,
#             step_size=0.1,
#             num_steps=20,
#             qnode_kwargs=qnode_kwargs,
#             verbose=True,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], -1, atol=atol)

#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-5), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     def test_optimize_mutual_info_matrix_separable_states(self, dev_kwargs, atol, qnode_kwargs):
#         qnp.random.seed(32)
#         opt_dict = qnetti.optimize_measured_mutual_info(
#             qnetvo.PrepareNode(wires=[0, 1]),
#             dev_kwargs=dev_kwargs,
#             step_size=0.1,
#             num_steps=12,
#             qnode_kwargs=qnode_kwargs,
#         )

#         assert qnp.isclose(opt_dict["min_cost"], 0, atol=atol)


# class TestOptimizeCharacteristicMatrix:
#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     @pytest.mark.parametrize("use_measured_mutual_info", [False, True])
#     def test_optimize_characteristic_matrix_two_qubit_ghz_state(
#         self,
#         dev_kwargs,
#         atol,
#         qnode_kwargs,
#         use_measured_mutual_info,
#     ):
#         prep_node = qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)
#         qnp.random.seed(55)

#         char_mat, mi_opt_dict, vn_opt_dict = qnetti.optimize_characteristic_matrix(
#             prep_node,
#             use_measured_mutual_info=use_measured_mutual_info,
#             dev_kwargs=dev_kwargs,
#             qnode_kwargs=qnode_kwargs,
#             mi_opt_kwargs={
#                 "step_size": 0.1,
#                 "num_steps": 15,
#             },
#             vn_opt_kwargs={
#                 "step_size": 0.1,
#                 "num_steps": 12,
#             },
#         )

#         assert qnp.allclose(char_mat, [[1, 1], [1, 1]], atol=atol)
#         assert len(mi_opt_dict["cost_vals"]) == 16
#         assert len(vn_opt_dict["cost_vals"]) == 13

#     @pytest.mark.parametrize("dev_kwargs, atol", [({}, 1e-4), ({"shots": 1000}, 1e-2)])
#     @pytest.mark.parametrize(
#         "qnode_kwargs",
#         [{}, {"diff_method": "parameter-shift"}],
#     )
#     @pytest.mark.parametrize("use_measured_mutual_info", [False, True])
#     def test_optimize_characteristic_matrix_separable_states(
#         self, dev_kwargs, atol, qnode_kwargs, use_measured_mutual_info
#     ):
#         qnp.random.seed(32)
#         char_mat, mi_opt_dict, vn_opt_dict = qnetti.optimize_characteristic_matrix(
#             qnetvo.PrepareNode(wires=[0, 1]),
#             use_measured_mutual_info=use_measured_mutual_info,
#             dev_kwargs=dev_kwargs,
#             qnode_kwargs=qnode_kwargs,
#             mi_opt_kwargs={
#                 "step_size": 0.1,
#                 "num_steps": 12,
#             },
#             vn_opt_kwargs={
#                 "step_size": 0.2,
#                 "num_steps": 25,
#             },
#         )

#         assert qnp.allclose(char_mat, qnp.zeros((2, 2)), atol=atol)
#         assert len(mi_opt_dict["cost_vals"]) == 13
#         assert len(vn_opt_dict["cost_vals"]) == 26

#     def test_optimize_characterisitic_matrix_measured_mutual_info_separations(
#         self, frustrated_mixed_state_prep_node
#     ):
#         qnp.random.seed(41)

#         char_mat1, _, _ = qnetti.optimize_characteristic_matrix(
#             frustrated_mixed_state_prep_node,
#             dev_kwargs={"name": "default.mixed"},
#             mi_opt_kwargs={"num_steps": 30, "step_size": 0.5},
#             use_measured_mutual_info=False,
#         )

#         assert qnp.allclose(char_mat1, [[1, 0, 0.188], [0, 1, 0], [0.188, 0, 1]], atol=1e-3)

#         char_mat2, _, _ = qnetti.optimize_characteristic_matrix(
#             frustrated_mixed_state_prep_node,
#             dev_kwargs={"name": "default.mixed"},
#             mi_opt_kwargs={"num_steps": 30, "step_size": 0.5},
#             use_measured_mutual_info=True,
#         )

#         assert qnp.allclose(
#             char_mat2, [[1, 0.188, 0.188], [0.188, 1, 0.188], [0.188, 0.188, 1]], atol=1e-3
#         )
