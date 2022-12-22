import qnetvo
import pennylane as qml


def qubit_characteristic_matrix(prepare_node):
    """Obtains the qubit characteristic matrix for a given multi-qubit state preparation.

    :param prepare_node: A state preparation for a multi-qubit state
    :type prepare_node: qnetvo.PrepareNode

    :returns:
    :rtype: matrix array
    """
    # TODO: obtain von neumann entropy of each qubit

    # TODO: obtain measured mutual information for each two-qubit pair

    return True
