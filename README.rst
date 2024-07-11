torch_solve_ext
---------------

This package is for solving differential equations, particularly,
for integrating ODEs that describe the flow of state variables.


We give an example of solving a system of ODEs,

.. math::

   \frac{dX}{dt} = f(X; \theta)

where :math:`X` is a vector variable that flows and :math:`\theta` is a set of
fixed parameters that control the flow. In addition to integrating the ODEs up
to a specific time, we would like to have :math:`\log|J|`, where :math:`J` is
the Jacobian matrix of the transformation. For back-propagation of derivatives,
we use the *adjoint* method. To do so, we need to define the flow function as a
subclass of ``FuncAdjWrapper``. Then we use ``AdjODEflow_`` to integrate the
ODE.

We first import the required modules:

.. code::

    import torch
    from torch_solve_ext.integrate import AdjODEflow_, FuncAdjWrapper


To be more specific, we focus on an ODE of the form

.. math::

   \frac{X}{dt} = f(x; \theta) = P @ (\sin(x) \circ \theta),

where :math:`[@, \circ]` denotes the matrix and element-wise products,
respectively. Here is the implementation (note that we assume that data has a
batch axis):

.. code::

    # projection matrix
    proj = - torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]) / 3
    proj = proj.unsqueeze(0)
    
    
    class Func(FuncAdjWrapper):
    
        @staticmethod
        def forward(t, var, frozen_var): 
            # t is irrelevant as the function is autonomous
            return proj @ (var.sin() * frozen_var)
    
        def calc_logj_rate(self, t, var, frozen_var):
            diag = torch.linalg.diagonal(proj)
            tr_df_dx = torch.sum(diag * (var.cos() * frozen_var).squeeze(-1), dim=1)
            return tr_df_dx


Note that the method ``calc_logj_rate`` is used for calculating the Jacobian of
transformation.

Let us now check the forward & reverse mode calculations

.. code::

    x = torch.randn(1, 3, 1)
    x -= torch.mean(x)
    param = torch.rand(1, 3, 1)
    print("X:\t", x.ravel())
    
    
    odeflow_ = AdjODEflow_(Func(), t_span=[0, 1], step_size=1e-2)
    
    y, logJ = odeflow_(x, param)
    print("f(X):\t", y.ravel())
    print("logJ:\t", logJ[0].item())
    
    z, logJ_r = odeflow_.reverse(y, param)
    
    print("X - f^{-1}(f(X)):", (x - z).ravel())
    print("logJ:", logJ + logJ_r)


.. parsed-literal::

    X:	 tensor([-0.6591,  0.2774,  0.3817])
    f(X):	 tensor([-0.4402,  0.2065,  0.2336])
    logJ:	 -0.6158064532779737
    X - f^{-1}(f(X)): tensor([ 4.7740e-15, -6.6613e-16, -4.0523e-15])
    logJ: tensor([-1.1102e-16])


One can follow the above example to define functions. Note that it is NOT
required to pass the parameters of the model as ``frozen_var``; one can pass
``None`` as ``frozen_var``.


| Created by Javad Komijani on 2024
| Copyright (c) 2024, Javad Komijani
