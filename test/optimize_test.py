import pytest

from pennylane import numpy as qnp

import qnetti


def test_optimize_one_param_quadratic():
    cost = lambda settings: settings[0] ** 2
    init_settings = qnp.array([1.0])

    opt_dict = qnetti.optimize(cost, init_settings, step_size=0.3, num_steps=10)

    assert opt_dict["step_size"] == 0.3

    assert len(opt_dict["cost_vals"]) == 11
    assert qnp.allclose(
        opt_dict["cost_vals"],
        [
            1.0,
            0.16,
            0.0256,
            0.004096,
            0.000655360,
            0.0001048576,
            1.6777216e-05,
            2.68435456e-06,
            4.294967296e-07,
            6.871947673e-08,
            1.09951162777e-08,
        ],
    )

    assert len(opt_dict["settings_list"]) == 11
    assert qnp.allclose(
        opt_dict["settings_list"],
        [
            [1.0],
            [0.4],
            [0.16],
            [0.064],
            [0.0256],
            [0.01024],
            [0.004096],
            [0.0016384],
            [0.00065536],
            [0.000262144],
            [0.0001048576],
        ],
    )

    assert qnp.isclose(opt_dict["min_cost"], 0, atol=1e-5)
    assert qnp.allclose(opt_dict["opt_settings"], [0], atol=2e-4)
