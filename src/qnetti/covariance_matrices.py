import numpy as np
import pennylane as qml
import qnetvo


def qubit_covariance_matrix_fn(prep_node, dev_kwargs={}, qnode_kwargs={}):
    """Generates a function that evaluates the covariance matrix for local
    qubit measurements.

    Each local qubit is measured in the :math:`z`-basis and is preced by an arbitrary
    qubit rotation as defined in PennyLane, |rot_ref|_.
    Using the joint probability distribution :math:`\{P(x_i)\}_i` constructed from the quantum circuit evaluation,
    we can evaluate the **covariance matrix** of an :math:`n`-qubit system as

    .. math::

        \text{Cov}(\{P(x_i)\}_{i}) = \begin{pmatrix}
            \text{Var}(x_1) &  \text{Cov}(x_1,x_2) & \dots &  \text{Cov}(x_1, x_n) \\
            \text{Cov}(x_2, x_1) & \text{Var}(x_2) & \dots & \text{Cov}(x_2, x_n) \\
            \vdots & &  \ddots & \vdots \\
            \text{Cov}(x_n, x_1) & \dots & \text{Cov}(x_n, x_{n-1} & \text{Var}(x_1, x_n) \\
        \end{pmatrix}
    
    where for two random variables :math:`x_i` and :math:`x_j`, the covariance is define
    :math:`\text{Cov}(x_i,x_j) = \langle (x_i - \langle x_i \rangle) (x_j - \langle x_j \rangle) \rangle`
    and the variance is defined :math:`\text{Var}(x_i) = \text{Cov}(x_i, x_i)`.
    Note that the covariance matrix is symmetric because :math:`\text{Cov}(x_i, x_j) = \text{Cov}(x_j, x_i)`.

    .. |rot_ref| replace:: ``qml.Rot()``
    .. _rot_ref: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.Rot.html?highlight=rot#pennylane.Rot

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function, ``covari
    :rtype: function
    
    """
    qubit_rot = lambda settings, wires: qml.Rot(*settings[:3], wires=wires)

    meas_nodes = [
        qnetvo.MeasureNode(1, 1, wires=[wire], quantum_fn=qubit_rot, num_settings=3)
        for wire in prep_node.wires
    ]

    ansatz = qnetvo.NetworkAnsatz([prep_node], meas_nodes, dev_kwargs=dev_kwargs)
    joint_probs = qnetvo.joint_probs_qnode(ansatz, **qnode_kwargs)

    return lambda meas_settings: qml.math.cov_matrix(
        joint_probs(meas_settings),
        [qml.PauliZ(wire) for wire in prep_node.wires],
        wires=qml.wires.Wires(prep_node.wires),
    )


def qubit_covariance_cost_fn(prep_node, dev_kwargs={}, qnode_kwargs={}):
    """Constructs a cost function that, when minimized, yields the maximal
    distance between the covariance matrix of the `prep_node` and the origin.

    That is, :math:`\text{Cost}(\Theta) = -\text{Tr}[\text{Cov}(\{P(x_i|\vec{\theta}_i)\}_i)^T \text{Cov}(\{P(x_i)\}_i)^T]`
    where the `meas_settings` are :math:`\Theta = (\vec{\theta}_i\in\mathbb{R^3)_{i=1}^n`.

    :param prep_node: A network node that prepares the quantum state to evaluate.
    :type prep_node: qnetvo.PrepareNode

    :param dev_kwargs: Keyword arguments passed to the PennyLane device constructor.
    :type dev_kwargs: dict

    :param qnode_kwargs: Keyword arguments passed to the PennyLane qnode constructor.
    :type qnode_kwargs: dict

    :returns: A function evaluated as ``cost(*meas_settings)``
    :rtype: function
    """

    cov_mat = qubit_covariance_matrix_fn(prep_node, dev_kwargs, qnode_kwargs)

    def qubit_covariance_cost(*meas_settings):
        mat = cov_mat(meas_settings)
        return -np.trace(mat.T @ mat)

    return qubit_covariance_cost
