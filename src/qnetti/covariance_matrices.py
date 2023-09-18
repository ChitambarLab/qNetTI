import pennylane as qml
from pennylane import numpy as qnp
import qnetvo

from .qnodes import qubit_probs_qnode_fn
from .optimize import optimize


def qubit_covariance_matrix_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Generates a function that evaluates the covariance matrix for local
    qubit measurements.

    Each local qubit is measured in the :math:`z`-basis and is preced by an arbitrary
    qubit rotation as defined in PennyLane, |rot_ref|_.
    Using the joint probability distribution :math:`\{P(x_i)\}_i` constructed from the quantum circuit evaluation,
    we can evaluate the **covariance matrix** of an :math:`n`-qubit system as

    .. math::

        \\text{Cov}(\\{P(x_i)\\}_{i}) = \\begin{pmatrix}
            \\text{Var}(x_1) &  \\text{Cov}(x_1,x_2) & \\dots &  \\text{Cov}(x_1, x_n) \\\\
            \\text{Cov}(x_2, x_1) & \\text{Var}(x_2) & \\dots & \\text{Cov}(x_2, x_n) \\\\
            \\vdots & &  \\ddots & \\vdots \\\\
            \\text{Cov}(x_n, x_1) & \\dots & \\text{Cov}(x_n, x_{n-1} & \\text{Var}(x_1, x_n) \\\\
        \\end{pmatrix}
    
    where for two random variables :math:`x_i` and :math:`x_j`, the covariance is define
    :math:`\\text{Cov}(x_i,x_j) = \\langle (x_i - \\langle x_i \\rangle) (x_j - \\langle x_j \\rangle) \\rangle`
    and the variance is defined :math:`\\text{Var}(x_i) = \\text{Cov}(x_i, x_i)`.
    Note that the covariance matrix is symmetric because :math:`\\text{Cov}(x_i, x_j) = \\text{Cov}(x_j, x_i)`.

    .. |rot_ref| replace:: ``qml.Rot()``
    .. _rot_ref: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.Rot.html?highlight=rot#pennylane.Rot

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits. 
    :type meas_wires: list[int]

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function, ``covariance_matrix(meas_settings)`` that takes as input a
              ``list[float]`` of length ``3 * num_wires``.
    :rtype: function
    """
    wires = meas_wires if meas_wires else prep_node.wires

    probs_qnode, dev = qubit_probs_qnode_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    return lambda meas_settings: qml.math.cov_matrix(
        probs_qnode(meas_settings),
        [qml.PauliZ(wire) for wire in wires],
        wires=qml.wires.Wires(wires),
    )


def qubit_covariance_cost_fn(prep_node, meas_wires=None, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a cost function that, when minimized, yields the maximal
    distance between the covariance matrix of the `prep_node` and the origin.

    That is, :math:`\\text{Cost}(\\Theta) = -\\text{Tr}[\\text{Cov}(\\{P(x_i|\\vec{\\theta}_i)\\}_i)^T \\text{Cov}(\\{P(x_i)\\}_i)^T]`
    where the ``meas_settings`` are :math:`\\Theta = (\\vec{\\theta}_i\\in\\mathbb{R}^3)_{i=1}^n`.

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param meas_wires: The wires to measure when evaluating the covariance matrix. If ``meas_wires`` are not specified,
                       all wires in the prepare node are considered. This can be used to ignore ancillary qubits.
    :type meas_wires: list[int]

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :returns: A function evaluated as ``cost(meas_settings)`` that takes as input a
              ``list[float]`` of length ``3 * num_wires``.
    :rtype: function
    """

    cov_mat = qubit_covariance_matrix_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    def qubit_covariance_cost(meas_settings):
        mat = cov_mat(meas_settings)
        return -qml.math.trace(mat.T @ mat)

    return qubit_covariance_cost


def optimize_covariance_matrix(
    prep_node,
    meas_wires=None,
    dev_kwargs={},
    qnode_kwargs={},
    **opt_kwargs,
):
    """Optimizes the arbitrary qubit measurements to maximize the distance between
    the covariance matrix and the origin.

    The optimization is performed using the :meth:`qnetti.optimize` method where
    the cost function is constructed using the :meth:`qubit_covariance_cost_fn` method.

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

    :returns: The first element of the returned tuple is the optimal covariance matrix while
              the second element is the dictionary returned from the :meth:`qnetti.optimize` method.
    :rtype: tuple[matrix, dict]
    """
    num_wires = len(meas_wires if meas_wires else prep_node.wires)
    init_settings = 2 * qnp.pi * qnp.random.rand(3 * num_wires, requires_grad=True)

    cov_cost = qubit_covariance_cost_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    opt_dict = optimize(
        cov_cost,
        init_settings,
        **opt_kwargs,
    )

    cov_mat_fn = qubit_covariance_matrix_fn(
        prep_node, meas_wires=meas_wires, dev_kwargs=dev_kwargs, qnode_kwargs=qnode_kwargs
    )

    cov_mat = cov_mat_fn(opt_dict["opt_settings"])

    return cov_mat, opt_dict
