import pennylane as qml
from pennylane import numpy as qnp
import time

from .file_utilities import datetime_now_string, tmp_dir, write_json


def optimize(
    cost,
    settings,
    step_size=0.1,
    num_steps=20,
    verbose=False,
    init_opt_dict=None,
    filepath="./",
    filename="",
    step_only=False,
    **meta_opt_kwargs,
):
    """Minimizes the ``cost`` function and find the optimal settings.

    The objective of the optimization is :math:`\\min_{\\vec{\\theta}} Cost(\\vec{\\theta})`.
    In the optimization a gradient descent algorithm is applied. In each iteration of gradient,
    descent, the settings :math:`\\vec{\\theta}`` are updated as

    .. math::

            \\vec{\\theta}' = \\vec{\\theta} - \\eta \\nabla_{\\vec{\\theta}} Cost(\\vec{\\theta})

    where :math:`\\eta` is the step size and :math:`\\nabla_{\\vec{\\theta}}Cost(\\vec{\\theta})` is
    the gradient of the cost function evaluated as :math:`\\vec{\\theta}`.

    :param cost: The function to be minimized. This function will be called as `cost(settings)`.
    :type cost: function

    :param settings: A PennyLane Numpy tensor that has the ``requires_grad=True`` attribute.
    :type settings: qml.numpy.ndarray

    :param step_size: The step to take in the direction of steepest descent. Default ``step_size=0.1``.
    :type step_size: float

    :param num_steps: The number of iterations of gradient descent to perform. Default ``num_steps=20``.
    :type num_steps: int

    :param verbose: If ``True``, the iteration step and cost will be printed every 5 iterations.
    :type verbose: bool

    :param init_opt_dict: An optimization dictionary to serve as a warm start for the optimization.
                          The optimization will continue from the provided dictionary.
    :type init_opt_dict: dictionary

    :param step_only: If ``True``, the cost is not evaluated. Default ``False``.
    :type step_only: bool

    :param meta_opt_kwargs: Generic keyword arguments to write the optimization dictionary.
    :type meta_opt_kwargs: keyword arguments

    :returns: A dictionary of optimization data.
    :rtype: dictionary

    The return dictionary has the following keys:
    * ``"settings_list"`` - ``list[list[float]]``, The settings considered in each optimization step.
    * ``"cost_vals"`` - ``list[float]``, The cost evaluated in each optimization step.
    * ``"min_cost"`` - ``float``, The smallest cost evaluated during optimization.
    * ``"opt_settings"`` - ``list[float]``, The optimal settings to achieve the minimum cost.
    * ``"step_size"`` - ``float``, The step size used in the optimization.
    * ``"datetime"`` - ``string``, The date/time at which the data is optimization begins.
    * ``"opt_step_time"`` - ``float``, The time needed to make each optimization step.
    * ``"error"`` - ``boolean``, Indicates whether or not an error occurred during optimization.
    """

    datetime_now_str = datetime_now_string()

    opt_dict = (
        init_opt_dict
        if init_opt_dict
        else {
            "settings_list": [settings.tolist()],
            "cost_vals": [],
            "step_size": step_size,
            "opt_step_times": [],
            "opt_settings": None,
            "min_cost": None,
            "datetime": datetime_now_str,
            "step_only": step_only,
            **meta_opt_kwargs,
        }
    )

    opt_dict["error"] = False

    opt = qml.GradientDescentOptimizer(stepsize=step_size)

    current_step = len(opt_dict["settings_list"]) - 1
    settings = qnp.array(opt_dict["settings_list"][-1], requires_grad=True)

    try:
        for i in range(current_step, num_steps):
            curr_time = time.time()

            if verbose:
                print("iteration : ", i)

            if step_only:
                settings = opt.step(cost, settings)
            else:
                settings, cost_val = opt.step_and_cost(cost, settings)
                opt_dict["cost_vals"] += [float(cost_val)]

                if verbose:
                    print("cost val : ", cost_val)

            opt_dict["settings_list"] += [settings.tolist()]
            opt_dict["opt_step_times"] += [time.time() - curr_time]
    except BaseException as err:
        msg_template = (
            "An exception of type {0} occurred during optimization step {1}. Arguments:\n{2!r}"
        )
        message = msg_template.format(type(err).__name__, str(i), err.args)
        print(message)

        opt_dict["error"] = True

        tmp_path = tmp_dir(filepath)
        write_json(opt_dict, tmp_path + filename + datetime_now_str)
    else:
        if not step_only:
            opt_dict["cost_vals"] += [float(cost(settings))]
            min_id = qml.math.argmin(opt_dict["cost_vals"])
            opt_dict["opt_settings"] = opt_dict["settings_list"][min_id]
            opt_dict["min_cost"] = opt_dict["cost_vals"][min_id]
        else:
            opt_dict["opt_settings"] = opt_dict["settings_list"][-1]

    return opt_dict
