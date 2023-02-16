import pennylane as qml
from pennylane import numpy as qnp

import qnetvo

from .qnodes import qubit_probs_qnode_fn
from .optimize import optimize


def qubit_shannon_entropies(probs_vec):
    """Given a probability distribution of qubit measurement data, evaluates
    the shannon entropy on each qubit subsystem.

    Let :math:`X` denote a random variable on the qubit measurement results. The Shannon
    entropy is defined as

    .. math::

        H(X) = - \\sum_i P(x_i) \\log_2(P(x_i))

    where :math:`P(x_i)` denotes the probability of the :math:`i^{th}` outcome.

    :param probs_vec: A probability distribution that contains positive elements that sum to one.
    :type probs_vec: np.array

    :returns: The :math:`i^{th}` list element corresponds to the Shannon entropy of the :math:`i^{th}` qubit.
    :rtype: list[float]
    """
    num_qubits = int(qml.math.log2(len(probs_vec)))

    probs_tensor = probs_vec.reshape((2,) * num_qubits)
    tensor_indices = "".join(chr(97 + q) for q in range(num_qubits))

    entropies = []
    for q1 in range(num_qubits):
        q1_index = chr(97 + q1)
        entropies += [
            qnetvo.shannon_entropy(qml.math.einsum(tensor_indices + "->" + q1_index, probs_tensor))
        ]

    return entropies


def qubit_mutual_infos(probs_vec):
    """Given a probability distribution of qubit measurement data, evaluates the
    mutual information between each pair of qubits.

    Let :math:`X` and :math:`Y` be random variables representing
    the measurement outcomes of two qubits in the network.
    The mutual information is then expressed as

    .. math::

            I(X;Y) = H(X) + H(Y) - H(XY)

    where :math:`H(\\cdot)` denotes the Shannon entropy.

    :param probs_vec: A probability distribution that contains positive elements that sum to one.
    :type probs_vec: np.array

    :returns: The mutual information between qubit pairs ``(q1, q2)`` where ``q1 + 1 <= q2``.
              The qubit pairs are ordered as ``(0,1), (0,2), ..., (1,2), (1,3), ...``.
    :rtype: list[float]
    """

    num_qubits = int(qml.math.log2(len(probs_vec)))

    probs_tensor = probs_vec.reshape((2,) * num_qubits)
    tensor_indices = "".join(chr(97 + q) for q in range(num_qubits))

    mutual_infos = []
    for q1 in range(num_qubits):
        q1_index = chr(97 + q1)
        for q2 in range(q1 + 1, num_qubits):
            q2_index = chr(97 + q2)

            HX = qnetvo.shannon_entropy(
                qml.math.einsum(tensor_indices + "->" + q1_index, probs_tensor)
            )

            HY = qnetvo.shannon_entropy(
                qml.math.einsum(tensor_indices + "->" + q2_index, probs_tensor)
            )

            HXY = qnetvo.shannon_entropy(
                qml.math.einsum(tensor_indices + "->" + q1_index + q2_index, probs_tensor).reshape(
                    (4)
                )
            )

            mutual_infos += [HX + HY - HXY]

    return mutual_infos


def shannon_entropy_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """
    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict
    """

    probs_qnode = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def shannon_entropy_cost(meas_settings):
        probs_vec = probs_qnode(meas_settings)

        return sum(qubit_shannon_entropies(probs_vec))

    return shannon_entropy_cost


def mutual_info_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict
    """

    probs_qnode = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def mutual_info_cost(meas_settings):
        probs_vec = probs_qnode(meas_settings)
        mutual_infos = qubit_mutual_infos(probs_vec)
        return -sum(mutual_infos)

    return mutual_info_cost


def qubit_characteristic_matrix_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """
    Given the preparation nodes, return a function that evaluates the characteristic matrix from two sets of settings,
    one for the Shannon entropies representing the diagonal elements, the other for the mutual information represeting
    the off-diagonal elements
    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict
    """

    probs_qnode = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def characteristic_matrix(vn_entropy_settings, mutual_info_settings):
        vn_entropy_probs = probs_qnode(vn_entropy_settings)
        mutual_info_probs = probs_qnode(mutual_info_settings)

        shannon_entropies = qubit_shannon_entropies(vn_entropy_probs)
        mutual_infos = qubit_mutual_infos(mutual_info_probs)
        num_qubits = len(shannon_entropies)

        char_mat = qml.math.zeros((num_qubits, num_qubits))

        char_mat[range(num_qubits), range(num_qubits)] = shannon_entropies

        id = 0
        for q1 in range(num_qubits):
            for q2 in range(q1 + 1, num_qubits):
                char_mat[(q1, q2), (q2, q1)] = mutual_infos[id]

                id += 1

        return char_mat

    return characteristic_matrix


def optimize_vn_entropy(
    prep_node,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    step_size=0.1,
    num_steps=10,
    verbose=False,
):
    num_wires = len(meas_wires if meas_wires else prep_node.wires)

    init_settings = 2 * qnp.pi * qnp.random.rand(3 * num_wires, requires_grad=True)
    shannon_entropy_cost = shannon_entropy_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return optimize(
        shannon_entropy_cost,
        init_settings,
        step_size=step_size,
        num_steps=num_steps,
        verbose=verbose,
    )


def optimize_mutual_info(
    prep_node,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    step_size=0.1,
    num_steps=10,
    verbose=False,
):
    num_wires = len(meas_wires if meas_wires else prep_node.wires)

    init_settings = 2 * qnp.pi * qnp.random.rand(3 * num_wires, requires_grad=True)
    mutual_info_cost = mutual_info_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return optimize(
        mutual_info_cost,
        init_settings,
        step_size=step_size,
        num_steps=num_steps,
        verbose=verbose,
    )


def optimize_characteristic_matrix(
    prep_node,
    cost_kwargs={},
    mi_opt_kwargs={},
    vn_opt_kwargs={},
):
    opt_mi_dict = optimize_mutual_info(
        prep_node,
        **cost_kwargs,
        **mi_opt_kwargs,
    )
    opt_vn_entropy_dict = optimize_vn_entropy(
        prep_node,
        **cost_kwargs,
        **vn_opt_kwargs,
    )

    char_mat = qubit_characteristic_matrix_fn(prep_node, **cost_kwargs)(
        opt_vn_entropy_dict["opt_settings"],
        opt_mi_dict["opt_settings"],
    )

    return char_mat, opt_mi_dict, opt_vn_entropy_dict
