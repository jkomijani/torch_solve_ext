# Copyright (c) 2024 Javad Komijani

from ._odeint import odeint

import torch
import numpy as np


def unitary_matrix_odeint(
        func, t_span, var0, frozen_var, step_size=1e-3,
        method='RK3:autonomous', loss_rate=None
        ):
    r"""Integrate a system of ODEs of the form:

    .. math::

        dV/dt = Z(t, V; p) V

    where :math:`V = V(t)` belongs to a Lie group and :math:`f(t, V; p)`
    belongs to corresponding algebra; see Appenix C of [`arXiv:1006.4518`_].
    Here ``p`` is used to denote fixed parameters that specify the flow
    (dynamic) of the system and ``t`` is a dummy variable when the system is
    autonomous.

    .. _arXiv:1006.4518: https://arxiv.org/abs/1006.4518

    Parameters
    ----------
    func: callable
        Computes :math:`Z(V)` given V and parameters.
    t_span: a sequence
        First and second items are the initial and terminal flow times.
    var0: tensor
        Initial state variable, i.e. the initial value of matrix ``V``.
    frozen_var: tensor
        The parameters specifying the system.
    step_size: float
        The absolute size of increament in the flow time ``t`` at each step
        (default is 0.001). The sign is determind by ``t_span``.
    method: str
        The integration method (default is 'RK3:autonomous').
    loss_rate: None or callable
        If callable, it will be treated as the integrand of a loss integral
        over time. It will be evaluated at each step, and summed over [with
        Simpson's rule], and the sum will be returned as output along with
        the state variable.
    """

    assert method == 'RK3:autonomous', "other methods are not implemented yet"

    step = unitary_matrix_autonomous_rk3_step

    return odeint(
            func, t_span, var0, frozen_var, step_size=step_size, method=step,
            loss_rate=loss_rate
            )


def unitary_matrix_autonomous_rk3_step(func, t, var, frozen_var, step_size):
    """Peforms one step of Runge-Kutta 3 method to integrate a system of ODEs
    of the form:

    .. math::

        dV/dt = f(t, V; p) V

    where :math:`V = V(t)` belongs to a Lie group and :math:`f(t, V; p)`
    belongs to corresponding algebra; see Appenix C of [`arXiv:1006.4518`_].
    Here ``p`` is used to denote fixed parameters that specify the flow
    (dynamic) of the system and ``t`` is a dummy variable when the system is
    autonomous.

    .. _arXiv:1006.4518: https://arxiv.org/abs/1006.4518
    """
    for ind in range(3):
        grad = func(t, var, frozen_var)  # f(t, V; p)
        # Below, zee = eps * Z defined in (C.1) & (1.4) of [arXiv:1006.4518]
        if ind == 0:
            zee = (1 / 4 * step_size) * grad
        elif ind == 1:
            zee = (8 / 9 * step_size) * grad - (17 / 9) * zee
        else:
            zee = (3 / 4 * step_size) * grad - zee
        var = torch.matrix_exp(zee) @ var
    return var
