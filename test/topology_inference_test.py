import pytest
import qnetvo
import pennylane as qml
import pennylane.numpy as np
import qnetti


def test_measured_mutual_info_cost_fn():
    def state_prep(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.Hadamard(wires=[wires[2]])
        qml.CNOT(wires=[wires[2], wires[3]])

    prep_node = qnetvo.PrepareNode(
        num_in=1, wires=[0, 1, 2, 3], ansatz_fn=state_prep, num_settings=0
    )

    # test entangled qubits
    meas_nodes = [
        qnetvo.MeasureNode(
            num_in=1, num_out=1, wires=[0], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
        ),
        qnetvo.MeasureNode(
            num_in=1, num_out=1, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
        ),
    ]
    cost_fn = qnetti.measured_mutual_info_cost_fn(qnetvo.NetworkAnsatz([prep_node], meas_nodes))

    assert np.isclose(cost_fn(*(0, 0, 0, 0, np.pi / 2, 0)), 0)
    assert np.isclose(cost_fn(*(0, 0, 0, 0, np.pi, np.pi)), -1)
    assert np.isclose(cost_fn(*(0, np.pi / 2, np.pi / 2, 0, np.pi / 2, np.pi / 2)), -1)

    # test unentangled qubits
    meas_nodes = [
        qnetvo.MeasureNode(
            num_in=1, num_out=1, wires=[1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
        ),
        qnetvo.MeasureNode(
            num_in=1, num_out=1, wires=[2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3
        ),
    ]
    cost_fn = qnetti.measured_mutual_info_cost_fn(qnetvo.NetworkAnsatz([prep_node], meas_nodes))

    assert np.isclose(cost_fn(*(0, 0, 0, 0, np.pi / 2, 0)), 0)
    assert np.isclose(cost_fn(*(0, 0, 0, 0, np.pi, np.pi)), 0)
    assert np.isclose(cost_fn(*(0, np.pi / 2, np.pi / 2, 0, np.pi / 2, np.pi / 2)), 0)


@pytest.mark.parametrize(
    "shots, qnode_kwargs, atol",
    [
        (None, {}, 1e-4),
        (100000, {}, 1e-4),
        (100000, {"diff_method": "parameter-shift"}, 1e-4),
    ],
)
def test_qubit_characteristic_matrix(shots, qnode_kwargs, atol):
    np.random.seed(101)

    # test that no entanglement sources are present
    def state_prep(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.Hadamard(wires=wires[1])
        qml.Hadamard(wires=wires[2])

    prep_node = qnetvo.PrepareNode(wires=[0, 1, 2], ansatz_fn=state_prep)
    assert np.allclose(
        qnetti.qubit_characteristic_matrix(
            prep_node, step_size=0.1, num_steps=25, shots=shots, qnode_kwargs=qnode_kwargs
        ),
        np.zeros(shape=(3, 3)),
        atol=atol,
    )


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
