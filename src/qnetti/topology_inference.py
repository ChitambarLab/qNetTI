import qnetvo
import pennylane as qml
import pennylane.numpy as np


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
