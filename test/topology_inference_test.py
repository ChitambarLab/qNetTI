import pytest
import qnetvo
import pennylane as qml

import qnetti


def test_qubit_characteristic_matrix():
    def state_prep(wires):
        qml.Hadamard(wires[0])
        qml.CNOT(wires[0:2])

        qml.Hadamard(wires[2])

    prep_node = qnetvo.PrepareNode(num_in=1, wires=[0, 1, 2], quantum_fn=state_prep, num_settings=0)

    # TODO: update tests
    assert qnetti.qubit_characteristic_matrix(prep_node)
