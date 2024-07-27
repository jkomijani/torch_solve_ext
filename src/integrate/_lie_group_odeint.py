# Copyright (c) 2024 Javad Komijani

from ._odeint import odeint
from ._adjoint import TupleVar

import torch


# =============================================================================
def lie_group_odeint(func, t_span, var0, frozen_var,
        step_size=1e-3, method='RK4:SU(n)', loss_rate=None
        ):
    r"""Integrate a system of ODEs of the form::

        dU / dt = f(t, U; p) = F(t, U; p) U

    where ``U = U(t)`` belongs to a Lie group and ``F(t, U; p)`` belongs to
    corresponding algebra. Here ``t`` is the flow time and ``p`` is used to
    denote fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func: callable
        Computes ``f(t, U; p)`` or ``F(t, U; p)``, depending on the ``method``.
        (Default is ``f``.)
    t_span: a sequence
        First and second items are the initial and terminal flow times.
    var0: tensor
        Initial state variable, i.e. the initial value of matrix ``U``.
    frozen_var: tensor
        The parameters specifying the system.
    step_size: float
        The absolute size of increament in the flow time ``t`` at each step
        (default is 0.001). The sign is determind by ``t_span``.
    method: str
        The integration method (default is 'RK4:SU(n)').
    loss_rate: None or callable
        If callable, it will be treated as the integrand of a loss integral
        over time. It will be evaluated at each step, and summed over [with
        Simpson's rule if possible], and the sum will be returned as output
        along with the state variable.
    """

    if method == 'RK4:SU(n)':
        step = special_unitary_rk4_step
    elif method == 'RK4:SU(n):aug':
        step = augmented_special_unitary_rk4_step
    elif method == 'RK3:u(n):auto':
        step = unitary_autonomous_rk3_step
    else:
        raise ValueError("other methods are not implemented yet")

    return odeint(
            func, t_span, var0, frozen_var, step_size=step_size, method=step,
            loss_rate=loss_rate
            )


# =============================================================================
def augmented_special_unitary_rk4_step(func, t, var, frozen_var, dt):
    delta, d_other = delta_from_rk4_step(func, t, var, frozen_var, dt).tuple
    var, other = var.tuple
    var = construct_rk4_special_unitary(delta @ var.adjoint()) @ var
    other = other + d_other
    return TupleVar(var, other)


def special_unitary_rk4_step(func, t, var, frozen_var, dt):
    """Perform a single Runge-Kutta-4 step for special unitary `var`.
    The output `var` is by contruction special unitary too.

    RK4 method gives

    .. math::

         U_{t + dt} = U_t + {shift} + O(h^5)= (I + \delta + O(h^5)) U_t

    We rewrite the coefficient of ``U_t`` as a special unitary matrix.
    """
    eps = delta_from_rk4_step(func, t, var, frozen_var, dt) @ var.adjoint()
    return construct_rk4_special_unitary(eps) @ var


def construct_rk4_special_unitary(eps):
    """Project ``I + \epsilon + O(\epsilon^5)`` to a special unitary matrix."""
    dummy = eps
    exponent = dummy
    for power in range(2, 5):
        # power of 5 and larger powers are not needed as error is O(eps^5)
        dummy = dummy @ (-eps)
        exponent = exponent + dummy / power
    return torch.matrix_exp(anti_hermitian_traceless(exponent))


def delta_from_rk4_step(func, t, var, frozen_var, dt):
    """Calculate the shift in var obtained from a single Runge-Kutta-4 step."""
    half_dt = dt / 2
    k_1 = func(t, var, frozen_var)
    k_2 = func(t + half_dt, var + half_dt * k_1, frozen_var)
    k_3 = func(t + half_dt, var + half_dt * k_2, frozen_var)
    k_4 = func(t + dt, var + dt * k_3, frozen_var)
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)


def anti_hermitian_traceless(mtrx):
    mtrx = (mtrx - mtrx.adjoint()) / 2.
    mu = torch.mean(torch.linalg.diagonal(mtrx), dim=-1, keepdim=True)
    mu = torch.diag_embed(torch.repeat_interleave(mu, mtrx.shape[-1], dim=-1))
    return mtrx - mu


# =============================================================================
def unitary_autonomous_rk3_step(algebra_func, t, var, frozen_var, step_size):
    """Peforms one step of Runge-Kutta 3 (?) method to integrate a system of
    integrate a system of ODEs of the form::

        dU / dt = F(t, U; p) U

    where ``U = U(t)`` belongs to a Lie group and ``F(t, U; p)`` belongs to
    corresponding algebra; see Appenix C of [`arXiv:1006.4518`_].

    Here ``p`` is used to denote fixed parameters that specify the flow
    (dynamic) of the system and ``t`` is a dummy variable when the system is
    autonomous.

    .. _arXiv:1006.4518: https://arxiv.org/abs/1006.4518
    """
    for ind in range(3):
        func_value = algebra_func(t, var, frozen_var)  # F(t, U; p)
        # Below, zee = eps * Z defined in (C.1) & (1.4) of [arXiv:1006.4518]
        if ind == 0:
            zee = (1 / 4 * step_size) * func_value
        elif ind == 1:
            zee = (8 / 9 * step_size) * func_value - (17 / 9) * zee
        else:
            zee = (3 / 4 * step_size) * func_value - zee
        var = torch.matrix_exp(zee) @ var
    return var
