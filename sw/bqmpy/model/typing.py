import collections.abc
import numpy as np
import numpy.typing as npt
import enum

__all__ = ['Variable', 'Bias', 'Vartype', 'SPIN', 'BINARY']

# Identifier for BQM variables
Variable = collections.abc.Hashable

# Quadratic, linear and offset bias values (any numeric value which is not complex)
Bias = int | float | np.integer[npt.NBitBase] | np.floating[npt.NBitBase]

# The variable-type (or encoding) of a BQM (SPIN->Ising, BINARY->QUBO)
class Vartype(enum.Enum):
    SPIN = frozenset({-1, 1}) # False, True
    BINARY = frozenset({0, 1}) # False, True

SPIN = Vartype.SPIN
BINARY = Vartype.BINARY
