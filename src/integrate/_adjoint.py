# Copyright (c) 2024-2025 Javad Komijani

"""
Defines the AdjODEflow_ module for adjoint-based ODE integration with
log-Jacobian tracking.
"""

import torch
import functools

from abc import abstractmethod, ABC

from ._odeint import odeint
from ..stats import hutchinson_estimator


# =============================================================================
class AdjODEflow_(torch.nn.Module):  # pylint: disable=invalid-name
    """
    A module for solving ODEs with adjoint-based backprop and log-Jacobian
    tracking.

    AdjODEflow_ is similar to `ODEflow_`, which extends `ODEflow` by also
    returning the log-determinant of the Jacobian of the flow. While `ODEflow`
    only evolves the state variable, `ODEflow_` and `AdjODEflow_` additionally
    compute the log-Jacobian of the transformation.

    This class uses the adjoint method to compute gradients during backward
    passes, which is memory efficient for long sequences.

    If `func` defines a method `calc_logj_rate`, it is used directly to compute
    the log-Jacobian rate. Otherwise, the trace of the Jacobian is estimated
    using the Hutchinson estimator with a given number of random samples.

    If `num_samples` is `None`, the Jacobian trace is computed exactly via
    automatic differentiation, which may be slow in high-dimensional settings.

    If `func` is not an instance of `DynamicsAdjModule`, it will automatically
    be wrapped with `DynamicsAdjWrapper`.

    Args:
        - func (Callable): A function `f(t, y, *args)` that computes the time
          derivative of the state variable `y` at time `t`.
        - t_span (Tuple[float, float]): A tuple specifying initial and final
          times.
        - num_samples (Optional[int | None]): The number of random samples used
          in the Hutchinson estimator. If `None`, the Jacobian trace is
          computed exactly. Defaults to 1.
        - **odeint_kwargs: Additional keyword arguments for the ODE solver.
    """

    def __init__(self, func, t_span, num_samples=1, **odeint_kwargs):
        """Initializes the AdjODEflow_ module."""

        super().__init__()
        if isinstance(func, DynamicsAdjModule):
            self.func = func
        else:
            self.func = DynamicsAdjWrapper(func, num_samples)
        self.t_span = t_span
        self.odeint = functools.partial(odeint, **odeint_kwargs)

    def forward(self, var, args=None, log0=0):
        """
        Evolves the state forward in time and accumulates the log-Jacobian.

        Args:
            var (torch.Tensor): Initial state variable to evolve.
            args (Optional[Tuple[torch.Tensor]]): Frozen variables of the
                dynamics (non-evolving parameters or context). Defaults to
                None.
            log0 (float): Initial value of the log-determinant of the Jacobian.
                Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The evolved state and the
            accumulated log-determinant of the Jacobian.
        """

        frozen_var = args

        params = self.func.params_

        var, logj = AdjointWrapper_.apply(
            self.odeint, self.func, self.t_span, var, frozen_var, *params
        )
        return var, logj + log0

    def reverse(self, var, args=None, log0=0):
        """
        Evolves the state backward in time and accumulates the log-Jacobian.

        Args:
            var (torch.Tensor): Final state variable to evolve backward.
            args (Optional[Tuple[torch.Tensor]]): Frozen variables of the
                dynamics (non-evolving parameters or context). Defaults to
                None.
            log0 (float): Initial value of the log-determinant of the Jacobian.
                Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The reversed state and the
            accumulated log-determinant of the Jacobian.
        """

        frozen_var = args

        params = self.func.params_

        var, logj = AdjointWrapper_.apply(
            self.odeint, self.func, self.t_span[::-1], var, frozen_var, *params
        )
        return var, logj + log0


# =============================================================================
class AdjointWrapper_(torch.autograd.Function):  # pylint: disable=invalid-name
    """
    A custom autograd Function to perform ODE integration using the adjoint
    method. This wraps around `odeint`, allowing gradients to flow through the
    integration.
    """

    @staticmethod
    def forward(ctx, odeint, func, t_span, var, frozen_var, *params):
        """
        Forward pass using the ODE solver.

        Args:
            ctx: Autograd context for saving information for backward pass.
            odeint: The ODE integration function.
            func: A DynamicsAdjModule instance defining the ODE system.
            t_span: Time span for integration.
            var: Initial state variable.
            frozen_var: Auxiliary variables, potentially requiring gradients.
            *params: Additional parameters to differentiate with respect to.

        Returns:
            var: The integrated state.
            logj: Accumulated log-Jacobian determinant, if applicable.

        NOTE:
            1. `frozen_var` must be a tensor if its gradient is needed.
            2. `*params` should be given explicitely in order to calculate the
            derivatives with respect to them.
        """
        assert isinstance(func, DynamicsAdjModule), \
            ("Expected `func` to be an instance of DynamicsAdjModule")

        # Perform ODE integration
        var, logj = odeint(
            func.forward,
            t_span,
            var,
            args=frozen_var,
            loss_rate=func.calc_logj_rate,
        )

        # Save necessary tensors for the backward pass
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

        # Define augmented variable, which will flow backward in time:
        aug_var = TupleVar(var, grad_var)
        # Define augmented frozen variable:
        aug_frozen_var = TupleVar(frozen_var, grad_logj, *params)

        fzn = frozen_var
        if len(params) == 0 and (fzn is None or not fzn.requires_grad):
            aug_var = odeint(
                func.aug_reverse, t_span[::-1], aug_var, args=aug_frozen_var
                )
            aug_loss = None
        else:
            aug_var, aug_loss = odeint(
                func.aug_reverse, t_span[::-1], aug_var, args=aug_frozen_var,
                loss_rate=func.calc_grad_params_rate
                )

        var, grad_var = aug_var.tuple

        if aug_loss is None:
            grad_frozen_var, grad_params = None, ()
        elif (fzn is None or not fzn.requires_grad):
            grad_frozen_var, grad_params = None, aug_loss.tuple
        else:
            grad_frozen_var, *grad_params = aug_loss.tuple

        return None, None, None, grad_var, grad_frozen_var, *grad_params


# =============================================================================
class DynamicsAdjModule(torch.nn.Module, ABC):
    """Any function that is used for defining the dynamics of an ODE flow with
    the adjoint method must be a subclass of this class.

    The method `calc_logj_rate` method calculates `Tr (df / dx)` is estimated
    using the Hutchinson estimator with a specified number of samples.
    """

    def __init__(self, num_samples=1):

        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def forward(self, t, var, *frozen_var):
        """The function defining the evolution of the state variable."""

    def calc_logj_rate(self, t, var, *frozen_var):
        """Return the trace of `df / dx` as the rate of `log(J)`."""
        logj_rate = hutchinson_estimator(
            lambda x: self.forward(t, x, *frozen_var),
            var,
            self.num_samples
        )
        return logj_rate

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


# =============================================================================
class DynamicsAdjWrapper(DynamicsAdjModule):

    def __init__(self, func, num_samples=1):

        super().__init__()
        self.func = func
        self.num_samples = num_samples

        if hasattr(func, 'calc_logj_rate'):
            self.calc_logj_rate = func.calc_logj_rate

    def forward(self, t, var, *args):
        """The function defining the evolution of the state variable."""
        return self.func(t, var, *args)


# =============================================================================
def tie_adjoints(x_bar, x_dot):
    r"""returns :math:`Re(\bar{x}^* \circ \dot{x})`,
    which is equal to :math:`ReTr(\bar{x}^\dagger \dot{x})`
    for matrix ``x`` in context of automtic differentiation,
    and :math:`\lambda^\dagger f` in context of the adjoint method.
    """
    dim = list(range(1, x_dot.ndim))
    return torch.sum((x_bar.conj() * x_dot).real, dim=dim)


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
        return tuple(getattr(var, "shape", 1) for var in self.tuple)
