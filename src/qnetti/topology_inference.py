import qnetvo
import pennylane as qml
import pennylane.numpy as np


def measured_mutual_info_cost_fn(ansatz, **qnode_kwargs):
    """Constructs an ansatz-specific measured mutual information cost function.

    In the context of quantum networks, the measured mutual information seeks to quantify the correlation between
    measurement statistics of a pair of measurement devices, where measurements are performed locally on each
    device.

    Formally, let :math:`X` and :math:`Y` be random variables representing measurement outcomes of two measurement
    devices in the network, where projective measurements :math:`\\{\\Pi^X\\}` and :math:`\\{\\Pi^Y\\}` are
    performed on respective devices. Then, the measured mutual information seeks to find the measurement bases
    that maximize the mutual information between :math:`X` and :math:`Y`:

    .. math::

            I_m (X;Y) = \\max_{\\{\\Pi^X\\}, \\{\\Pi^X\\}} H(X) + H(Y) - H(XY)

    where :math:`H(\cdot)` denotes the Shannon entropy.

    :param ansatz: The ansatz circuit on which the measured mutual information is evaluated. The ansatz is limited
                   to having two measurement nodes.
    :type ansatz: qnetvo.Network Ansatz

    :param qnode_kwargs: Keyword arguments passed to execute qnodes.
    :type qnode_kwargs: dictionary

    :return: A cost function ``measured_mutual_info_cost(*network_settings)`` parameterized by the ansatz-specific
             scenario settings.
    :rtype: Function

    """
    num_prep_nodes = len(ansatz.layers[0])
    num_meas_nodes = len(ansatz.layers[-1])
    num_qubits = len(ansatz.layers_wires[-1])
    X_num_qubits = len(ansatz.layers[-1][0].wires)
    Y_num_qubits = len(ansatz.layers[-1][1].wires)

    probs_qnode = qnetvo.joint_probs_qnode(ansatz, **qnode_kwargs)

    def cost(*network_settings):
        settings = ansatz.qnode_settings(
            network_settings, [[0] * num_prep_nodes, [0] * num_meas_nodes]
        )
        probs_vec = probs_qnode(settings)

        probs_tensor = probs_vec.reshape((2,) * (X_num_qubits + Y_num_qubits))

        HX = qnetvo.shannon_entropy(
            np.sum(probs_tensor, axis=tuple(range(X_num_qubits, num_qubits))).reshape(
                (2 * X_num_qubits),
            )
        )
        HY = qnetvo.shannon_entropy(
            np.sum(probs_tensor, axis=tuple(range(X_num_qubits))).reshape((2 * Y_num_qubits,))
        )

        HXY = qnetvo.shannon_entropy(probs_vec)

        return -(HX + HY - HXY)

    return cost


def qubit_characteristic_matrix(
    prepare_node, step_size=0.05, num_steps=501, shots=None, qnode_kwargs={}
):
    """Obtains the qubit characteristic matrix for a given multi-qubit state preparation.

    Mathematically, the qubit characteristic matrix is a real-valued matrix :math:`Q \\in \\mathbb{R}^{n \times n}`,
    :math:`n` being the number of qubits in a network. On the diagonal, :math:`Q` stores the von Neumann entropy of
    the respective qubit, i.e. for any :math:`i \\in [n]`, :math:`Q_{ii} = S(q_i)`. On the other hand, off-diagonal
    entries stores the measured mutual information between qubits: :math:`Q_{ij} = I_m(q_i;q_j)` for :math:`i \neq j`.
    For further details, see https://arxiv.org/abs/2212.07987.

    :param prepare_node: A state preparation for a multi-qubit state
    :type prepare_node: qnetvo.PrepareNode

    :param step_size: The magnitude of each step during optimization
    :type step_size: float

    :param num_steps: The number of iterations to perform the optimization
    :type num_steps: int

    :param shots: the number of measurement evaluated on the quantum circuit
    :type shots: int

    :returns: the calculated/measured characteristic matrix of input network
    :rtype: matrix array
    """

    num_qubits = max([qubit for qubit in prepare_node.wires]) + 1
    if shots is None:
        device = {"name": "default.mixed"}
    else:
        device = {"name": "default.mixed", "shots": shots}

    characteristic_matrix = np.zeros(shape=[num_qubits, num_qubits])
    # compute von Neumann entropies
    for q in range(num_qubits):
        meas_node = qnetvo.MeasureNode(wires=[q], ansatz_fn=qml.ArbitraryUnitary, num_settings=3)
        ansatz = qnetvo.NetworkAnsatz([prepare_node], [meas_node], dev_kwargs=device)
        cost = qnetvo.shannon_entropy_cost_fn(ansatz, **qnode_kwargs)
        settings = ansatz.rand_network_settings()
        dict = qnetvo.gradient_descent(
            cost,
            settings,
            step_size=step_size,
            sample_width=1,
            num_steps=num_steps,
            verbose=False,
        )

        characteristic_matrix[q, q] = -max(dict["scores"])

    # compute meausured mutual information
    for q1 in range(num_qubits):
        for q2 in range(q1 + 1, num_qubits):
            meas_nodes = [
                qnetvo.MeasureNode(wires=[q1], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
                qnetvo.MeasureNode(wires=[q2], ansatz_fn=qml.ArbitraryUnitary, num_settings=3),
            ]
            ansatz = qnetvo.NetworkAnsatz([prepare_node], meas_nodes, dev_kwargs=device)
            cost = measured_mutual_info_cost_fn(ansatz, **qnode_kwargs)
            settings = ansatz.rand_network_settings()
            dict = qnetvo.gradient_descent(
                cost,
                settings,
                step_size=step_size,
                sample_width=1,
                num_steps=num_steps,
                verbose=False,
            )
            characteristic_matrix[q1, q2] = max(dict["scores"])
            characteristic_matrix[q2, q1] = max(dict["scores"])

    return characteristic_matrix


def characteristic_matrix_decoder(characteristic_matrix, tol=1e-5):
    """Decode the qubit characteristic matrix and partition qubits into their respective preparation nodes.

    If two qubits come from the same preparation node, they are correlated and have identical rows in the qubit
    characteristic matrix.

    :param characteristic_matrix: the qubit characteristic matrix of an unknown network
    :type characteristic_matrix: matrix array

    :param tol: tolerance for distinguishing non-zero elements
    :type: float

    :returns: a list of lists representing qubits that shares entanglement sources
    :rtype: list[list[int]]
    """

    num_qubits = len(characteristic_matrix)

    # network is a dictionary with prep. nodes as keys,
    # and the list of qubits that belong to the respective prep node as values
    network = {}

    # convert characteristic matrix to binary (zero/non-zero)
    characteristic_matrix = np.where(np.abs(characteristic_matrix) > tol, 1, 0)

    for qubit in range(num_qubits):
        prep_node = "".join(map(str, characteristic_matrix[qubit, :]))
        if prep_node in network:
            qubit_list = network[prep_node]
            qubit_list.append(qubit)
            network[prep_node] = qubit_list
        else:
            network[prep_node] = [qubit]

    return [network[prep_node] for prep_node in network]
