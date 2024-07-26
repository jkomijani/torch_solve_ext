# Copyright (c) 2024 Javad Komijani

import torch
import functools

from abc import abstractmethod, ABC

from ._odeint import odeint


# =============================================================================
class AdjODEflow_(torch.nn.Module):
    """A `Module` for evolution of state variables governed by ODEs.

    Similar to `ODEflow`, but it also returns the logarithm of Jacobian of
    transformation provided that `func` has `calc_logj_rate` attribute.
    More importantly, this class provides a backward propagation of derivatives
    using the adjoint method. To this end, `func` is required be a subclass of
    `DynamcisAdjWrapper`.

    See `ODEflow` for description of the class.
    """
    # TODO: change such that `func` need not be a subclass of `DynamcisAdjWrapper`.

    def __init__(self, func, t_span, **odeint_kwargs):

        assert isinstance(func, DynamcisAdjWrapper)

        super().__init__()
        self.func = func
        self.t_span = t_span
        self.odeint = functools.partial(odeint, **odeint_kwargs)
    
    def forward(self, var, frozen_var=None):
        params = self.func.params_
        var, logj = AdjointWrapper_.apply(
            self.odeint, self.func, self.t_span, var, frozen_var, *params
            )
        return var, logj

    def reverse(self, var, frozen_var=None):
        params = self.func.params_
        var, logj = AdjointWrapper_.apply(
            self.odeint, self.func, self.t_span[::-1], var, frozen_var, *params
            )
        return var, logj


# =============================================================================
class AdjointWrapper_(torch.autograd.Function):
    """A wrapper to run ``odeint`` with the adjoint method."""

    # Because of the convention adopted by pytorch, ``frozen_var`` must be a
    # tensor if its `grad` is meant to be returned, otherwise, it must be given
    # as a tuple of tensors.

    @staticmethod
    def forward(ctx, odeint, func, t_span, var, frozen_var, *params):
        # `*params` should be given in order to calculate derivatives wrt them.

        assert isinstance(func, DynamcisAdjWrapper)

        var, logj = odeint(
                func.forward, t_span, var, frozen_var,
                loss_rate=func.calc_logj_rate
                )

        ctx.odeint = odeint
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
        odeint = ctx.odeint
        t_span = ctx.t_span
        var, logj, frozen_var, *params = ctx.saved_tensors

        # Define augmented variables & flow them in backward in time:
        aug_var = TupleVar(var, grad_var)
        aug_frozen_var = TupleVar(frozen_var, grad_logj, *params)

        aug_var, aug_loss = odeint(
                func.aug_reverse, t_span[::-1], aug_var, aug_frozen_var,
                loss_rate=func.calc_grad_params_rate
                )
        var, grad_var = aug_var.tuple

        if (frozen_var is None) or (not frozen_var.requires_grad):
            grad_frozen_var, grad_params = None, aug_loss.tuple
        else:
            grad_frozen_var, *grad_params = aug_loss.tuple

        return None, None, None, grad_var, grad_frozen_var, *grad_params


# =============================================================================
class DynamcisAdjWrapper(torch.nn.Module, ABC):
    """Any function that is used for defining the dynamics of an ODE flow with
    the adjoint method must be a subclass of this class.

    The dynamics governing the flow of state variable is supposed to be defined
    in `forward`, which is an abstract method should be defined in subclasses.
    Although `calc_logj_rate` method calculates ``Tr (df / dx)``, it is in
    general slow, and we recommend to redefine it in subclasses.
    """

    @abstractmethod
    def forward(self, t, var, frozen_var=None):
        """The function defining the evolution of the state variable."""
        pass

    def calc_logj_rate(self, t, var, frozen_var=None):
        """Return the trace of ``df / dx`` as the rate of ``log(J)``."""
        func = lambda var: self.forward(t, var, frozen_var)
        return trace_df_dx(func, var)

    def aug_reverse(self, t, aug_var, aug_frozen_var):
        """Here `aug_var` contains the state variable and the adjoint state
        variable at time `t`. Similarly, `aug_frozen_var` contains the frozen
        variables, `grad_logj`, defined as gradient of the loss function w.r.t.
        `logj`, and the parameters of the model.
        """
        var, grad_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        with torch.enable_grad():
            var = var.detach().requires_grad_(True)

            var_dot = self.forward(t, var, frozen_var)
            logj_dot = self.calc_logj_rate(t, var, frozen_var)
            hamilton = torch.sum(
                    grad_logj * logj_dot + tie_adjoints(grad_var, var_dot)
                    )

            grad_var_dot, = \
                torch.autograd.grad(- hamilton, (var,), retain_graph=False)

        return TupleVar(var_dot, grad_var_dot)

    def calc_grad_params_rate(self, t, aug_var, aug_frozen_var):
        var, grad_var = aug_var.tuple
        frozen_var, grad_logj, *params = aug_frozen_var.tuple

        if (frozen_var is None) or (not frozen_var.requires_grad):
            params = tuple(self.params_)
        else:
            frozen_var = frozen_var.detach().requires_grad_(True)
            params = (frozen_var, *self.params_)

        with torch.enable_grad():
            var = var.detach()

            var_dot = self.forward(t, var, frozen_var)
            logj_dot = self.calc_logj_rate(t, var, frozen_var)
            hamilton = torch.sum(
                    grad_logj * logj_dot + tie_adjoints(grad_var, var_dot)
                    )

            grad_params_rate = torch.autograd.grad(
                    - hamilton, params,
                    retain_graph=False, materialize_grads=True
                    )

        return TupleVar(*grad_params_rate)

    @property
    def params_(self):
        return [par for par in self.parameters() if par.requires_grad]


def tie_adjoints(x_bar, x_dot):
    r"""returns :math:`Re(\bar{x}^* \circ \dot{x})`,
    which is equal to :math:`ReTr(\bar{x}^\dagger \dot{x})`
    for matrix ``x`` in context of automtic differentiation,
    and :math:`\lambda^\dagger f` in context of the adjoint method.
    """
    dim = list(range(1, x_dot.ndim))
    return torch.sum((x_bar.conj() * x_dot).real, dim=dim)


def trace_df_dx(func, var):
    """Calculates the trace of ``df(x) / dx`` for real variables."""

    if torch.is_complex(var):
        return trace_complex_df_dx(func, var)

    # The following manipulations is for using torch.autograd.grad
    if not var.requires_grad:
        var = var.detach().requires_grad_(True)
    x = var.reshape(var.shape[0], -1)
    f = func(x.reshape(*var.shape)).sum(dim=0).reshape(-1)

    trace = 0.
    for ind in range(x.shape[1]):
        trace += torch.autograd.grad(f[ind], x, retain_graph=True)[0][:, ind]

    return trace


def trace_complex_df_dx(func, var):
    """Calculates the trace of ``df(x) / dx`` for complex variables."""

    # The following manipulations is for using torch.autograd.grad
    if not var.requires_grad:
        var = var.detach().requires_grad_(True)
    x = var.reshape(var.shape[0], -1).real
    y = var.reshape(var.shape[0], -1).imag
    f = func((x + 1j * y).reshape(*var.shape)).sum(dim=0).reshape(-1)

    trace = 0.
    for k in range(x.shape[1]):
        trace += torch.autograd.grad(f[k].real, x, retain_graph=True)[0][:, k]
        trace += torch.autograd.grad(f[k].imag, y, retain_graph=True)[0][:, k]

    return trace


# =============================================================================
class TupleVar:

    def __init__(self, *args):
        self.tuple = args

    def __str__(self):
        return f"TupleVar:\n{self.tuple}"

    def __repr__(self):
        return self.__str__()

    def __pos__(self):
        return self

    def __neg__(self):
        return TupleVar(*[-var for var in self.tuple])

    def __add__(self, other):
        x = [var1 + var2 for var1, var2 in zip(self.tuple, other.tuple)]
        return TupleVar(*x)

    def __sub__(self, other):
        x = [var1 - var2 for var1, var2 in zip(self.tuple, other.tuple)]
        return TupleVar(*x)

    def __mul__(self, other):
        return TupleVar(*[var * other for var in self.tuple])

    def __truediv__(self, other):
        return TupleVar(*[var / other for var in self.tuple])

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def shape(self):
        def shape(v):
            try: return v.shape
            except: return 1
        return tuple([shape(var) for var in self.tuple])
