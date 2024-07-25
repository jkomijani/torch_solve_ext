# Copyright (c) 2024 Javad Komijani

from ._lie_group_odeint import lie_group_odeint
from ._adjoint import TupleVar
from ._adjoint import tie_adjoints
from ._adjoint import trace_complex_df_dx

import torch
from functools import partial as ftpartial

from abc import abstractmethod, ABC


# =============================================================================
class LieGroupAdjODEflow_:
    """For flowing state variables governed by a system of ODEs."""

    def __init__(self, func, t_span,
            methods=['RK4:SU(n)', 'RK4:SU(n):aug'], **odeint_kwargs
            ):
        super().__init__()
        assert isinstance(func, LieGroupFuncAdjWrapper)
        self.func = func
        self.t_span = t_span
        self.odeints = [
            ftpartial(lie_group_odeint, method=methods[0], **odeint_kwargs),
            ftpartial(lie_group_odeint, method=methods[1], **odeint_kwargs)
            ]

    def __call__(self, var, frozen_var):
        return self.forward(var, frozen_var)

    def forward(self, var, frozen_var):
        params = self.func.params_
        var, logj = LieGroupAdjointWrapper_.apply(
           self.odeints, self.func, self.t_span, var, frozen_var, *params
           )
        return var, logj

    def reverse(self, var, frozen_var):
        params = self.func.params_
        var, logj = LieGroupAdjointWrapper_.apply(
           self.odeints, self.func, self.t_span[::-1], var, frozen_var, *params
           )
        return var, logj


# =============================================================================
class LieGroupAdjointWrapper_(torch.autograd.Function):
    """A wrapper to run ``odeint`` with the adjoint method."""

    # Because of the convention adopted by pytorch, ``frozen_var`` must be a
    # tensor if its `grad` is meant to be returned, otherwise, it must be given
    # as a tuple of tensors.

    @staticmethod
    def forward(ctx, odeints, func, t_span, var, frozen_var, *params):
        # `*params` should be given in order to calculate derivatives wrt them.

        assert isinstance(func, LieGroupFuncAdjWrapper)

        var, logj = odeints[0](
                func.forward, t_span, var, frozen_var,
                loss_rate=func.calc_logj_rate
                )

        ctx.odeints = odeints
        ctx.func = func
        ctx.t_span = t_span
        ctx.save_for_backward(var, logj, frozen_var, *params)

        return var, logj

    @staticmethod
    def backward(ctx, grad_var, grad_logj):
        """Evaluated by integrating the augmented system backwards in time."""
        # grad_{var, logj} are $\bar var$ and $\bar logj$ in terminology of AD
        # grad_{var} is $\lambda$ in the terminology of adjoint method
        # grad_{var} input/output are the $\lambda$ at terminal/initial times

        func = ctx.func
        odeints = ctx.odeints
        t_span = ctx.t_span
        var, logj, frozen_var, *params = ctx.saved_tensors

        # Define augmented variables & flow them backward in time:
        # Instead of calculating `grad_var`, for lie group, we first calculate
        # `grad_alg_var`, which is the `grad` fo the algebra variable
        # corresponding to `var` and in the end we convert it to `grad_var`.
        grad_alg_var = grad_var @ var.adjoint()

        aug_var = TupleVar(var, grad_alg_var)
        aug_frozen_var = TupleVar(frozen_var, grad_logj, *params)

        aug_var, aug_loss = odeints[1](
                func.aug_reverse, t_span[::-1], aug_var, aug_frozen_var,
                loss_rate=func.calc_grad_params_rate
                )
        var, grad_alg_var = aug_var.tuple
        # grad_alg_var must be already anti-hermitian, yet we project it again.
        grad_var = anti_hermitian(grad_alg_var) @ var

        if (frozen_var is None) or (not frozen_var.requires_grad):
            grad_frozen_var, grad_params = None, aug_loss.tuple
        else:
            grad_frozen_var, *grad_params = aug_loss.tuple

        return None, None, None, grad_var, grad_frozen_var, *grad_params


def anti_hermitian(mtrx):
    return (mtrx - mtrx.adjoint()) / 2.


# =============================================================================
class LieGroupFuncAdjWrapper(torch.nn.Module, ABC):
    """Any function that is used for defining the dynamics of an ODE flow with
    the adjoint method for lie groups must be a subclass of this class.

    The dynamics governing the algebra variable is supposed to be defined in
    `algebra_dynamics`, which is an abstract method should be defined in
    subclasses.
    Although `calc_logj_rate` method calculates ``Re Tr (df / dx)``, it is in
    general slow, and we recommend to redefine it in subclasses.

    Integrate a system of ODEs of the form::

        dU / dt = F(t, U; p) U

    where ``U = U(t)`` belongs to a Lie group and ``F(t, U; p)`` belongs to
    corresponding algebra. Here ``t`` is the flow time and ``p`` is used to
    denote fixed parameters that specify the flow (dynamic) of the system.

    One should define ``algbra_dynamics`` that returns ``F(t, U; p)``.
    """

    def forward(self, t, var, frozen_var):
        """The function defining the evolution of the state variable."""
        return self.algebra_dynamics(t, var, frozen_var) @ var

    @abstractmethod
    def algebra_dynamics(self, t, var, frozen_var):
        """The function defining ``F(t, U; p)``."""
        pass

    def calc_logj_rate(self, t, var, frozen_var):
        """Return ``Re Tr (df / dx)`` as the rate of ``log(J)``."""
        func = lambda var: self.forward(t, var, frozen_var)
        return trace_complex_df_dx(func, var)

    def aug_reverse(self, t, aug_var, aug_frozen_var):
        """Here `aug_var` contains the state variable and the adjoint state
        variable at time `t`. Similarly, `aug_frozen_var` contains the frozen
        variables, `grad_logj`, defined as gradient of the loss function w.r.t.
        `logj`, and the parameters of the model.
        """
        var, grad_alg_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        with torch.enable_grad():
            var = var.detach().requires_grad_(True)

            alg_var_dot = self.algebra_dynamics(t, var, frozen_var)
            logj_dot = self.calc_logj_rate(t, var, frozen_var)
            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
                )

            grad_var_dot, = \
                torch.autograd.grad(-hamilton, (var,), retain_graph=False)
        # Note that: ``var_dot = alg_var_dot @ var``
        # & ``grad_alg_var_dot = grad_var_dot @ var.adjoint()``

        return TupleVar(alg_var_dot @ var, grad_var_dot @ var.adjoint())

    def calc_grad_params_rate(self, t, aug_var, aug_frozen_var):
        var, grad_alg_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        if (frozen_var is None) or (not frozen_var.requires_grad):
            params = tuple(self.params_)
        else:
            frozen_var = frozen_var.detach().requires_grad_(True)
            params = (frozen_var, *self.params_)

        with torch.enable_grad():
            var = var.detach()

            alg_var_dot = self.algebra_dynamics(t, var, frozen_var)
            logj_dot = self.calc_logj_rate(t, var, frozen_var)
            hamilton = torch.sum(
                grad_logj * logj_dot + tie_adjoints(grad_alg_var, alg_var_dot)
                )

            grad_params_rate = torch.autograd.grad(
                    -hamilton, params,
                    retain_graph=False, materialize_grads=True
                    )

        return TupleVar(*grad_params_rate)

    @property
    def params_(self):
        return [par for par in self.parameters() if par.requires_grad]
