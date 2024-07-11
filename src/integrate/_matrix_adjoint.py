# Copyright (c) 2024 Javad Komijani

from ._adjoint import AdjODEflow_
from ._matrix_odeint import unitary_matrix_odeint


class UnitaryMatrixAdjODEflow_(AdjODEflow_):
    cls_odeint = (unitary_matrix_odeint,)
