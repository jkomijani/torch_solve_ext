# Copyright (c) 2024 Javad Komijani

from ._lie_group_odeint import lie_group_odeint

import torch
import functools


class LieGroupODEflow(torch.nn.Module):
    """A `Module` for evolution of Lie-group state variables governed by ODEs.

    Given a derivative function ``f``, an instance of this class flows the
    state of the system according to::

        dU / dt = f(t, U; p) = F(t, U; p) U

    where ``U = U(t)`` belongs to a Lie group and ``F(t, U; p)`` belongs to
    corresponding algebra. Here ``t`` is the flow time and ``p`` is used to
    denote fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func: callable
        Computes ``f(t, U; p)`` or ``F(t, U; p)``; see ``lie_group_odeint``.
        (Default is ``f``.)
    t_span: list/tuple
        First and second items are the initial and terminal flow times.
    odeint_kwargs: optional kwargs
        Keyword arguments to be passed to the ODE integrator.
    """

    def __init__(self, func, t_span, **odeint_kwargs):
        super().__init__()
        self.func = func
        self.t_span = t_span
        self.odeint = functools.partial(lie_group_odeint, **odeint_kwargs)

    def forward(self, var, args=None):
        return self.odeint(self.func, self.t_span, var, args=args)

    def reverse(self, var, args=None):
        return self.odeint(self.func, self.t_span[::-1], var, args=args)


class LieGroupODEflow_(LieGroupODEflow):
    """A `Module` for evolution of Lie-group state variables governed by ODEs.

    Similar to `LieGroupODEflow`, but it also returns the logarithm of Jacobian
    of transformation provided that `func` has `calc_logj_rate` attribute.

    See `LieGroupODEflow` for description of the class.
    """

    def __init__(self, func, t_span, **odeint_kwargs):

        assert hasattr(func, 'calc_logj_rate')

        super().__init__(
                func, t_span, loss_rate=func.calc_logj_rate, **odeint_kwargs
                )
