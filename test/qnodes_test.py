import pytest
import numpy as np
import qnetvo
import pennylane as qml

import qnetti


@pytest.mark.parametrize(
    "prep_node, meas_wires, dev_kwargs, settings, probs_match",
    [
        (
            qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state),
            None,
            {},
            [0, 0, 0, 0, 0, 0],
            [0.5, 0, 0, 0.5],
        ),
        (
            qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state),
            None,
            {},
            [np.pi / 2, 0, 0, 0, 0, 0],
            [0.25, 0.25, 0.25, 0.25],
        ),
        (
            qnetvo.PrepareNode(wires=[0, 1, 2], ansatz_fn=qnetvo.ghz_state),
            None,
            {},
            [0, 0, 0, 0, 0, 0, np.pi / 2, 0, 0],
            [0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25],
        ),
        (
            qnetvo.PrepareNode(wires=[0, 1, 2], ansatz_fn=qnetvo.ghz_state),
            [0, 1],
            {},
            [0, 0, 0, 0, 0, 0],
            [0.5, 0, 0, 0.5],
        ),
        (
            qnetvo.PrepareNode(
                wires=[0, 1],
                ansatz_fn=lambda settings, wires: qml.DepolarizingChannel(0.5, wires=[0]),
            ),
            None,
            {"name": "default.mixed"},
            [0, 0, 0, 0, 0, 0],
            [2 / 3, 0, 1 / 3, 0],
        ),
    ],
)
def test_qubit_probs_qnode_fn(prep_node, meas_wires, dev_kwargs, settings, probs_match):
    probs_qnode, dev = qnetti.qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs
    )

    assert np.allclose(probs_qnode(settings), probs_match)
    assert dev.short_name == dev_kwargs["name"] if "name" in dev_kwargs else "default.qubit"
