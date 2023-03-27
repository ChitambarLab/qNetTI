import pennylane as qml


def qubit_probs_qnode_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a qnode function that returns the probabilities for local qubit measurements.
    The resulting quantum circuit prepares the state described by the ``prep_node`` and then
    applies arbitrary qubit unitaries to each wire before measurement.

    :param prep_node: An ansatz for the network state preparation.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to apply arbitrary qubit measurements to. If ``None``, all qubits are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments to pass to the PennyLane device constructor. Useful keys are
                       include ``"name"``, which specifies the device (default: ``"default.qubit"``), and ``"shots"``,
                       which specifies the integert number of shots to run during circuit execution (default: ``None``).
    :type shots: dict

    :param qnode_kwargs: Keyword arguments passed through to the PennyLane qnode decorator.
    :type qnode_kwargs: dict

    :returns: A qnode function that returns the qubit probabilities for the circuit and the device
              that evaluates the qnode.
              The function is called as ``qnode(settings)`` where ``len(settings) == 3 * num_wires``.
    :rtype: tuple(qml.QNode, qml.Device)
    """

    meas_wires = meas_wires if meas_wires else prep_node.wires

    dev_kwargs_copy = dev_kwargs.copy()
    dev_name = dev_kwargs_copy.pop("name", "default.qubit")

    dev = qml.device(dev_name, wires=prep_node.wires, **dev_kwargs_copy)

    @qml.qnode(dev, **qnode_kwargs)
    def qubit_probs_qnode(settings):
        prep_node([])
        for i, wire in enumerate(meas_wires):
            qml.ArbitraryUnitary(settings[3 * i : 3 * i + 3], wires=[wire])

        return qml.probs(wires=meas_wires)

    return qubit_probs_qnode, dev
