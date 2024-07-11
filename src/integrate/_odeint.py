# Copyright (c) 2024 Javad Komijani

import numpy as np


def odeint(func, t_span, var0, frozen_var, step_size=1e-3, method='RK4',
        loss_rate=None
        ):
    r"""Integrate a system of ODEs.

    The system of ODEs is::

         dy / dt = f(t, y; p)

    where ``t`` is the flow time of the system, ``y`` is the state variable
    (a vector describing the state of the system) at time ``t``, and ``p`` is
    a set of fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func: callable
        Computes the derivative of vector ``y`` at time ``t``.
    t_span: a sequence
        First and second items are the initial and terminal flow times.
    var0: tensor
        Initial state variable, i.e. the initial value of vector ``y``.
    frozen_var: tensor
        The parameters specifying the system.
    step_size: float
        The size of increament in the flow time ``t`` at each step
        (default is 0.001). The sign is determind by ``t_span``. Note that the
        given `step_size` is only tentative and depending on the initial and
        terminal flow times, the exact `step_size` may change slightly.
    method: str or callable
        The integration method as a string or a function (default is 'RK4').
    loss_rate: None or callable
        If callable, it will be treated as the integrand of a loss integral
        over time. It will be evaluated at each step, and summed over [with
        Simpson's rule], and the sum will be returned as output along with
        the state variable.

    Remark::
    This code can handle any number of batch axes if ``func`` can handle them.
    """
    if hasattr(method, '__call__'):
        step = method
    elif method == 'RK4':
        step = rk4_step
    elif method == 'Euler':
        step = euler_step
    else:
        raise Exception("other methods are not implemented yet")

    n_grid = 1 + 2 * abs(int((t_span[1] - t_span[0]) / step_size / 2))
    # n_grid is hard wired to be odd in order to use Simpson's formula.
    t_range = np.linspace(*t_span, n_grid)
    step_size = t_range[1] - t_range[0]  # slightly modified step_size

    var = var0

    if loss_rate is None:
        for t in enumerate(t_range[:-1]):
            var = step(func, t, var, frozen_var, step_size)
        return var

    else:
        assert hasattr(loss_rate, '__call__')
        loss = loss_rate(t_range[0], var, frozen_var)
        for ind, t in enumerate(t_range[:-1]):
            var = step(func, t, var, frozen_var, step_size)
            l_r = loss_rate(t_range[ind + 1], var, frozen_var)
            loss += l_r * (4 if ind % 2 == 0 else 2)
        loss -= l_r  # for the last part
        loss *= (step_size / 3)
        return var, loss


def euler_step(func, t, var, frozen_var, dt):
    """Perform a single Euler step."""
    return var + func(t, var, frozen_var) * dt


def rk4_step(func, t, var, frozen_var, dt):
    """Perform a single Runge-Kutta step."""
    half_dt = dt / 2
    k_1 = func(t, var, frozen_var)
    k_2 = func(t + half_dt, var + half_dt * k_1, frozen_var)
    k_3 = func(t + half_dt, var + half_dt * k_2, frozen_var)
    k_4 = func(t + dt, var + dt * k_3, frozen_var)
    return var + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)


def autonomous_rk4(func, var0, frozen_var, step_size=1e-4, num_steps=1):
    r"""Integrate an `autonomous` system of ODEs using Runge-Kutta 4 method."""
    eps = step_size
    var = var0
    for n in range(num_steps):
        k_1 = func(var, frozen_var)
        k_2 = func(var + eps / 2 * k_1, frozen_var)
        k_3 = func(var + eps / 2 * k_2, frozen_var)
        k_4 = func(var + eps * k_3, frozen_var)
        var = var + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (eps / 6)
    return var
