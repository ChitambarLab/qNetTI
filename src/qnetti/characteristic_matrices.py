import pennylane as qml
from pennylane import numpy as qnp
import math

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


def qubit_measured_mutual_infos_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a function that evaluates the mutual information for all qubit pairs in the
    network where each qubit pair is measured using unique settings.

    The mutual information is computed for each qubit pair using :meth:`qnetti.qubit_mutual_infos`
    on the two-qubit probability distribution output from the qubit measurements.

    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The qubit wires to measure in the network. If ``None`` all wires are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function ``qubit_measured_mutual_infos(settings)`` where settings has length 6 times the
              number of distinct qubit pairs.
    :rtype: function
    """

    wires = meas_wires if meas_wires else prep_node.wires

    probs_qnodes = []
    for i, q1 in enumerate(wires):
        for q2 in wires[i + 1 :]:
            probs_qnode, dev = qubit_probs_qnode_fn(
                prep_node, meas_wires=[q1, q2], dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
            )
            probs_qnodes += [probs_qnode]

    def qubit_measured_mutual_infos(settings):
        measured_mutual_infos = []
        for i, qnode in enumerate(probs_qnodes):
            probs_vec = qnode(settings[6 * i : 6 * i + 6])
            measured_mutual_infos += qubit_mutual_infos(probs_vec)

        return measured_mutual_infos

    return qubit_measured_mutual_infos


def shannon_entropy_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a cost function from the sum of Shannon entropoes on each qubit.

    Let :math:`X_i` denote the random variable associated with the :math:`i^{th}` qubit's
    measurement result. The Shannon entropy cost function is then

    .. math::

        Cost = \\sum_{i} H(X_i)

    where the Shannon entropy :math:`H(X_i)` is evaluated in :meth:`qnetti.qubit_shannon_entropies`.

    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The qubit wires to measure in the network. If ``None`` all wires are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function ``shannon_entropy_cost(settings)`` that evaluates the specified cost function.
    :rtype: qml.QNode
    """

    probs_qnode, dev = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def shannon_entropy_cost(meas_settings):
        probs_vec = probs_qnode(meas_settings)

        return sum(qubit_shannon_entropies(probs_vec))

    return shannon_entropy_cost


def mutual_info_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a cost function from the sum of mutual information across all unique pairs
    of qubits in the network.

    Let :math:`X_i` denote the random variable associated with the :math:`i^{th}` qubit's
    measurement result. The mutual information cost function is then

    .. math::

        Cost = -\\sum_{i < j} I(X_i ; X_j)

    where the mutual information :math:`I(X_i ; X_j)` is evaluated in :meth:`qnetti.qubit_mutual_infos`

    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The qubit wires to measure in the network. If ``None`` all wires are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function ``mutual_info_cost(settings)`` that evaluates the specified cost function.
    :rtype: qml.QNode
    """

    probs_qnode, dev = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def mutual_info_cost(settings):
        probs_vec = probs_qnode(settings)
        mutual_infos = qubit_mutual_infos(probs_vec)
        return -sum(mutual_infos)

    return mutual_info_cost


def measured_mutual_info_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a cost function for maximizing the measured mutual information across all qubit
    pairs in the network.

    In the context of quantum networks, the measured mutual information quantifies the correlation between
    measurement statistics of a pair of qubit measurement devices that each perform a local measurement.

    Formally, let :math:`X` and :math:`Y` be random variables representing measurement outcomes of two measurement
    devices in the network, where projective measurements :math:`\\{\\Pi^X\\}` and :math:`\\{\\Pi^Y\\}` are
    performed on respective devices. Then, the measured mutual information seeks to find the measurement bases
    that maximize the mutual information between :math:`X` and :math:`Y`:

    .. math::

            I_m (X:Y) = \\max_{\\{\\Pi^X\\}, \\{\\Pi^X\\}} H(X) + H(Y) - H(XY)

    where :math:`H(\cdot)` denotes the Shannon entropy.
    The returned cost function is then

    .. math::

            Cost = -\\sum_{i<j} I_m(X_i:Y_j).

    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The qubit wires to measure in the network. If ``None`` all wires are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function ``measured_mutual_info_cost(settings)`` that evaluates the specified cost function.
    :rtype: Function
    """

    qubit_measured_mutual_infos = qubit_measured_mutual_infos_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def measured_mutual_info_cost(settings):
        measured_mutual_infos = qubit_measured_mutual_infos(settings)
        return -sum(measured_mutual_infos)

    return measured_mutual_info_cost


def qubit_characteristic_matrix_fn(
    prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}, use_measured_mutual_info=False
):
    """Given the preparation nodes, return a function that evaluates the characteristic matrix from two sets of settings,
    one for the Shannon entropies representing the diagonal elements, the other for the mutual information represeting
    the off-diagonal elements

    :param prep_node: a network node that prepares the quantum state to evaluate
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The qubit wires to measure in the network. If ``None`` all wires are measured.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param use_measured_mutual_info: If ``True`` the measured mutual information is evaluated with respect to
                                     all qubit pairs. Default ``False``.
    :type use_measured_mutual_info: bool

    :returns: A function ``characteristic_matrix(vn_entropy_settings, mutual_info_settings)`` where the
              parameters are the qubit measurement settings for the von Neumann entropy and mutual information.
    :rtype: function
    """

    probs_qnode, dev = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    qubit_measured_mutual_infos = qubit_measured_mutual_infos_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def characteristic_matrix(vn_entropy_settings, mutual_info_settings):
        vn_entropy_probs = probs_qnode(vn_entropy_settings)
        shannon_entropies = qubit_shannon_entropies(vn_entropy_probs)
        num_qubits = len(shannon_entropies)

        mutual_infos = (
            qubit_measured_mutual_infos(mutual_info_settings)
            if use_measured_mutual_info
            else qubit_mutual_infos(probs_qnode(mutual_info_settings))
        )

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
    **opt_kwargs,
):
    """Optimizes the network's arbitrary qubit measurements to minimize the :meth:`qnetti.shannon_entropy_cost_fn`.
    The minimum Shannon entropy corresponds to the von Neumann entropy.

    The optimization is performed using the :meth:`qnetti.optimize` method where

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param step_size: The step to take in the direction of steepest descent. Default ``step_size=0.1``.
    :type step_size: float

    :param num_steps: The number of iterations of gradient descent to perform. Default ``num_steps=20``.
    :type num_steps: int

    :param verbose: If ``True``, the iteration step and cost will be printed every 5 iterations.
    :type verbose: bool

    :returns: The dictionary returned from :meth:`qnetti.optimize`
    :rtype: dict
    """
    num_wires = len(meas_wires if meas_wires else prep_node.wires)

    init_settings = 2 * qnp.pi * qnp.random.rand(3 * num_wires, requires_grad=True)
    shannon_entropy_cost = shannon_entropy_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return optimize(
        shannon_entropy_cost,
        init_settings,
        **opt_kwargs,
    )


def optimize_mutual_info(
    prep_node,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    **opt_kwargs,
):
    """Optimizes the network's arbitrary qubit measurements to minimize the :meth:`qnetti.mutual_info_cost_fn`.
    See the :meth:`qnetti.optimize` method for details regarding the gradient optimization.

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param step_size: The step to take in the direction of steepest descent. Default ``step_size=0.1``.
    :type step_size: float

    :param num_steps: The number of iterations of gradient descent to perform. Default ``num_steps=20``.
    :type num_steps: int

    :param verbose: If ``True``, the iteration step and cost will be printed every 5 iterations.
    :type verbose: bool

    :returns: The dictionary returned from :meth:`qnetti.optimize`
    :rtype: dict
    """
    num_wires = len(meas_wires if meas_wires else prep_node.wires)

    init_settings = 2 * qnp.pi * qnp.random.rand(3 * num_wires, requires_grad=True)
    mutual_info_cost = mutual_info_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return optimize(
        mutual_info_cost,
        init_settings,
        **opt_kwargs,
    )


def optimize_measured_mutual_info(
    prep_node,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    **opt_kwargs,
):
    """Optimizes the network's arbitrary qubit measurements to minimize the :meth:`qnetti.measured_mutual_info_cost_fn`.
    See the :meth:`qnetti.optimize` method for details regarding the gradient optimization.

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param step_size: The step to take in the direction of steepest descent. Default ``step_size=0.1``.
    :type step_size: float

    :param num_steps: The number of iterations of gradient descent to perform. Default ``num_steps=20``.
    :type num_steps: int

    :param verbose: If ``True``, the iteration step and cost will be printed every 5 iterations.
    :type verbose: bool

    :returns: The dictionary returned from :meth:`qnetti.optimize`
    :rtype: dict
    """

    num_wires = len(meas_wires if meas_wires else prep_node.wires)

    init_settings = 2 * qnp.pi * qnp.random.rand(6 * math.comb(num_wires, 2), requires_grad=True)
    mutual_info_cost = measured_mutual_info_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return optimize(mutual_info_cost, init_settings, **opt_kwargs)


def optimize_characteristic_matrix(
    prep_node,
    use_measured_mutual_info=False,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    mi_opt_kwargs={},
    vn_opt_kwargs={},
):
    """Obtains the qubit characteristic matrix for a given multi-qubit state preparation.

    Mathematically, the qubit characteristic matrix is a real-valued matrix :math:`Q \\in \\mathbb{R}^{n \\times n}`,
    :math:`n` being the number of qubits in a network. On the diagonal, :math:`Q` stores the von Neumann entropy of
    the respective qubit, i.e. for any :math:`i \\in [n]`, :math:`Q_{ii} = S(q_i)`. On the other hand, off-diagonal
    entries stores the measured mutual information between qubits: :math:`Q_{ij} = I_m(q_i;q_j)` for :math:`i \\neq j`.
    For further details, see https://arxiv.org/abs/2212.07987.

    This function uses the :meth:`qnetti.optimize_vn_entropy` and :meth:`qnetti.optimize_mutual_info` methods to
    find the optimal measurement settinggs for inferring the network's topology.

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits.
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param step_size: The step to take in the direction of steepest descent. Default ``step_size=0.1``.
    :type step_size: float

    :param num_steps: The number of iterations of gradient descent to perform. Default ``num_steps=20``.
    :type num_steps: int

    :param verbose: If ``True``, the iteration step and cost will be printed every 5 iterations.
    :type verbose: bool

    :param use_measured_mutual_info: If ``True`` the measured mutual information is evaluated with respect to
                                     all qubit pairs. Default ``False``.
    :type use_measured_mutual_info: bool

    :returns: A tuple containing the characteristic matrix, the mutual information optimizaation resullts,
              and the von Neumann entropy optimization results.
    :rtype: tuple[matrix, dict, dict]
    """

    cost_kwargs = {
        "meas_wires": meas_wires,
        "dev_kwargs": dev_kwargs,
        "qnode_kwargs": qnode_kwargs,
    }

    opt_mi_dict = (
        optimize_measured_mutual_info(
            prep_node,
            **cost_kwargs,
            **mi_opt_kwargs,
        )
        if use_measured_mutual_info
        else optimize_mutual_info(
            prep_node,
            **cost_kwargs,
            **mi_opt_kwargs,
        )
    )
    opt_vn_entropy_dict = optimize_vn_entropy(
        prep_node,
        **cost_kwargs,
        **vn_opt_kwargs,
    )

    char_mat = qubit_characteristic_matrix_fn(
        prep_node,
        use_measured_mutual_info=use_measured_mutual_info,
        **cost_kwargs,
    )(
        opt_vn_entropy_dict["opt_settings"],
        opt_mi_dict["opt_settings"],
    )

    return char_mat, opt_mi_dict, opt_vn_entropy_dict
