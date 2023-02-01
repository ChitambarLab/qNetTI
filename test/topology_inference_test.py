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


def test_qubit_characteristic_matrix():
    np.random.seed(102)

    # test that no entanglement sources are present
    def state_prep(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.Hadamard(wires=wires[1])
        qml.Hadamard(wires=wires[2])

    prep_node = qnetvo.PrepareNode(num_in=1, wires=[0, 1, 2], ansatz_fn=state_prep, num_settings=0)
    assert np.allclose(
        qnetti.qubit_characteristic_matrix(prep_node, step_size=0.1, num_steps=25),
        np.zeros(shape=(3, 3)),
        atol=1e-4,
    )

    np.random.seed(151)

    # test for one entanglement source
    def state_prep(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CNOT(wires=[wires[1], wires[2]])

    prep_node = qnetvo.PrepareNode(num_in=1, wires=[0, 1, 2], ansatz_fn=state_prep, num_settings=0)
    assert np.allclose(
        qnetti.qubit_characteristic_matrix(prep_node, step_size=0.1, num_steps=25),
        np.ones(shape=(3, 3)),
        atol=1e-4,
    )

    # test for two preparation nodes
    def state_prep(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.Hadamard(wires=[wires[2]])
        qml.CNOT(wires=[wires[2], wires[3]])

    prep_node = qnetvo.PrepareNode(
        num_in=1, wires=[0, 1, 2, 3], ansatz_fn=state_prep, num_settings=0
    )
    assert np.allclose(
        qnetti.qubit_characteristic_matrix(prep_node, step_size=0.1, num_steps=25),
        np.kron(np.eye(2), np.ones(shape=(2, 2))),
        atol=1e-4,
    )


def test_characteristic_matrix_decoder():
    # test one prep node
    assert qnetti.characteristic_matrix_decoder(np.zeros(shape=(3, 3))) == [[0, 1, 2]]
    assert qnetti.characteristic_matrix_decoder(np.ones(shape=(3, 3))) == [[0, 1, 2]]

    # test for multiple prep node
    assert qnetti.characteristic_matrix_decoder(np.kron(np.eye(2), np.ones(shape=(2, 2)))) == [
        [0, 1],
        [2, 3],
    ]
    assert qnetti.characteristic_matrix_decoder(
        np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
    ) == [[0, 2], [1, 3]]
