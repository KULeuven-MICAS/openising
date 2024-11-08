import enum
from collections.abc import Hashable

__all__ = ['Variable', 'Bias', 'Vartype']

# Identifier for BQM variables
Variable = Hashable

# Quadratic, linear and offset bias values (any numeric value which is not complex)
Bias = int | float

# The variable-type (or encoding) of a BQM
class Vartype(enum.Enum):
    SPIN = frozenset({-1, 1}) # False, True
    BINARY = frozenset({0, 1}) # False, True
