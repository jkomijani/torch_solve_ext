# Copyright (c) 2024 Javad Komijani

from ._odeint import odeint

import torch
import functools


class ODEflow(torch.nn.Module):
    """A `Module` for evolution of state variables governed by ODEs.

    Given a derivative function ``f``, an instance of this class flows the
    state of the system according to::

         dy / dt = f(t, y; p)

    where ``t`` denotes the flow time of the system, ``y`` is a vector
    describing the state of the system at time ``t``, and ``p`` is a set of
    fixed parameters that specify the flow (dynamic) of the system. The fixed
    parameters can either be embeded in the definition of ``f`` or can be
    explicitely given as `args`.

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
        super().__init__()
        self.func = func
        self.t_span = t_span
        self.odeint = functools.partial(odeint, **odeint_kwargs)

    def forward(self, var, args=None):
        return self.odeint(self.func, self.t_span, var, args=args)

    def reverse(self, var, args=None):
        return self.odeint(self.func, self.t_span[::-1], var, args=args)


class ODEflow_(ODEflow):
    """A `Module` for evolution of state variables governed by ODEs.

    Similar to `ODEflow`, but it also returns the logarithm of Jacobian of
    transformation provided that `func` has `calc_logj_rate` attribute.

    See `ODEflow` for description of the class.
    """

    def __init__(self, func, t_span, **odeint_kwargs):

        assert hasattr(func, 'calc_logj_rate')

        super().__init__(
                func, t_span, loss_rate=func.calc_logj_rate, **odeint_kwargs
                )
