import qnetvo as qnet
import pennylane as qml
import pennylane.numpy as np


def measured_mutual_info_cost_fn(ansatz, **qnode_kwargs):
    num_prep_nodes = len(ansatz.prepare_nodes)
    num_meas_nodes = len(ansatz.measure_nodes)
    num_qubits = len(ansatz.measure_wires)
    X_num_qubits = len(ansatz.measure_nodes[0].wires)
    Y_num_qubits = len(ansatz.measure_nodes[1].wires)

    probs_qnode = qnet.joint_probs_qnode(ansatz, **qnode_kwargs)

    def cost(*network_settings):
        settings = ansatz.qnode_settings(
            network_settings, [[0] * num_prep_nodes, [0] * num_meas_nodes]
        )
        probs_vec = probs_qnode(settings)
        probs_tensor = probs_vec.reshape((2,) * (X_num_qubits + Y_num_qubits))

        HX = qnet.shannon_entropy(
            np.sum(probs_tensor, axis=tuple(range(X_num_qubits, num_qubits))).reshape(
                (2 * X_num_qubits),
            )
        )
        HY = qnet.shannon_entropy(
            np.sum(probs_tensor, axis=tuple(range(X_num_qubits))).reshape((2 * Y_num_qubits,))
        )

        HXY = qnet.shannon_entropy(probs_vec)

        return -(HX + HY - HXY)

    return cost


def qubit_characteristic_matrix(prepare_node, step_size=0.05, num_steps=101, shot=0):
    """Obtains the qubit characteristic matrix for a given multi-qubit state preparation.

    :param prepare_node: A state preparation for a multi-qubit state
    :type prepare_node: qnetvo.PrepareNode

    :param step_size: The magnitude of each step during optimization
    :type step_size: float

    :param num_step: The number of iterations to perform the optimization
    :type num_steps: int

    :param shot: the number of measurement taken on the quantum state
    :type shot: int

    :returns: the calculated/measured characteristic matrix of input network
    :rtype: matrix array
    """

    num_qubit = max([qubit for qubit in prepare_node.wires]) + 1
    if shot > 0:
        device = {"name": "default.qubit", "shots": 10}
    else:
        device = {}

    characteristic_matrix = np.zeros(shape=[num_qubit, num_qubit])
    # compute von Neumann entropies
    for qubit in range(num_qubit):
        meas_node = [
            qnet.MeasureNode(1, 1, wires=[qubit], quantum_fn=qml.ArbitraryUnitary, num_settings=3)
        ]
        ansatz = qnet.NetworkAnsatz([prepare_node], meas_node, dev_kwargs=device)
        cost = qnet.shannon_entropy_cost_fn(ansatz)
        settings = ansatz.rand_network_settings()
        dict = qnet.gradient_descent(
            cost,
            settings,
            step_size=step_size,
            sample_width=num_steps - 1,
            num_steps=num_steps,
            verbose=False,
        )
        characteristic_matrix[qubit, qubit] = -dict["scores"][-1]

    # compute meausured mutual information
    for qubit_1 in range(num_qubit):
        for qubit_2 in range(qubit_1 + 1, num_qubit):
            meas_node = [
                qnet.MeasureNode(
                    1, 1, wires=[qubit_1], quantum_fn=qml.ArbitraryUnitary, num_settings=3
                ),
                qnet.MeasureNode(
                    1, 1, wires=[qubit_2], quantum_fn=qml.ArbitraryUnitary, num_settings=3
                ),
            ]
            ansatz = qnet.NetworkAnsatz([prepare_node], meas_node)
            cost = measured_mutual_info_cost_fn(ansatz)
            settings = ansatz.rand_network_settings()
            dict = qnet.gradient_descent(
                cost,
                settings,
                step_size=step_size,
                sample_width=num_steps - 1,
                num_steps=num_steps,
                verbose=False,
            )
            characteristic_matrix[qubit_1, qubit_2] = dict["scores"][-1]
            characteristic_matrix[qubit_2, qubit_1] = dict["scores"][-1]

    return characteristic_matrix


def characteristic_matrix_decoder(characteristic_matrix, tol=1e-5):
    """Decode the qubit characteristic matrix by grouping indices that have the same rows

    :param characteristic_matrix: the qubit characteristic matrix of an unknown network
    :type characteristic_matrix: matrix array

    :param tol: tolerance for distinguishing non-zero elements
    :type: float

    :returns: a list of lists representing qubits that shares entanglement sources
    :rtype: list
    """

    dict = {}
    num_qubit = characteristic_matrix.shape[0]

    # convert characteristic matrix to binary (zero/non-zero)
    characteristic_matrix = np.where(characteristic_matrix > tol, 1, 0)

    def array_to_string(arr):
        dim = arr.shape[0]
        string = ""
        for i in range(dim):
            string += str(arr[i])
        return string

    for row in range(num_qubit):
        key = array_to_string(characteristic_matrix[row, :])
        if key in dict:
            val = dict[key]
            val.append(row)
            dict[key] = val
        else:
            dict[key] = [row]

    return [dict[key] for key in dict]
