# Copyright (c) 2024 Javad Komijani

from ._odeint import odeint

import functools


class ODEflow:
    r"""For evolution of the state of systems governed by a system of ODEs.

    Given a derivative function ``f``, an instance of this class flows the
    state of a system according to::

         dy / dt = f(t, y; p)

    where ``t`` denotes the flow time of the system, ``y`` is a vector
    describing the state of the system at time ``t``, and ``p`` is a set of
    fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func: callable
        Computes the derivative of vector ``y`` at time ``t``.
    t_span: list/tuple
        First and second items are the initial and terminal flow times.
    odeint_kwargs: optional kwargs
        Keyword arguments to be passed to the ODE integrator.
    """

    def __init__(self, func, t_span, **odeint_kwargs):
        self.func = func
        self.t_span = t_span
        self.odeint = functools.partial(odeint, **odeint_kwargs)
    
    def __call__(self, var, frozen_var):
        return self.forward(var, frozen_var)

    def forward(self, var, frozen_var):
        return self.odeint(self.func, self.t_span, var, frozen_var)

    def reverse(self, var, frozen_var):
        return self.odeint(self.func, self.t_span[::-1], var, frozen_var)


class ODEflow_(ODEflow):

    def __init__(self, func, t_span, **odeint_kwargs):
        super().__init__(
                func, t_span, loss_rate=func.calc_logj_rate, **odeint_kwargs
                )
