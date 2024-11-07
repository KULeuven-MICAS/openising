from __future__ import annotations
import numpy as np
from collections.abc import Collection, Sequence
from pathlib import Path
from ising.typing import Variable, Bias, Vartype
from ising.utils.convert import LinearLike, convert_to_linear, QuadraticLike, convert_to_quadratic

__all__ = ['BinaryQuadraticModel']


class BinaryQuadraticModel:
    """Encodes a binary quadratic model.

    Attributes:
        linear (dict[Variable, Bias]):
            A dictionary mapping variables in the BQM to its linear bias. 
            Biasless variables have bias zero.

        quadratic (dict[frozenset[Variable, Variable], Bias]):
            A dictionary mapping unordered collections of 2 variables (edges/pairs in
            the BQM) to their quadratic biases. Omitted edges are implicitly zero.
            Every variable present here should have an entry in self.linear.

        offset (Bias):
            Offset of the BQM. This constant becomes meaningless in optimization solving
            but is a necessary part for proper QUBO-Ising mapping.

        vartype (Vartype):
            Encoding type of the BQM, either Vartype.SPIN or Vartype.BINARY.

    """

    __slots__ = ['linear', 'quadratic', 'offset', 'vartype']

    def __init__(self, linear: LinearLike, quadratic: QuadraticLike, offset: Bias, vartype: Vartype):
        if not isinstance(vartype, Vartype):
            raise ValueError(f'Vartype unkown: {vartype}')
        self.linear: dict[Variable, Bias] = {}
        self.quadratic: dict[frozenset[Variable, Variable], Bias] = {}
        self.offset: Bias = 0
        self.vartype: Vartype = vartype

        self.set_variables_from(linear)
        self.set_interactions_from(quadratic)
        self.set_offset(offset)

    def __repr__(self) -> str:
        return f'BinaryQuadraticModel({self.linear}, {self.quadratic}, {self.offset}, {self.vartype})'

    def __len__(self) -> int:
        return self.num_variables

    def __eq__(self, other) -> bool:
        if not isinstance(other, BinaryQuadraticModel):
            return False
        return all([
            self.linear == other.linear,
            self.quadratic == other.quadratic,
            self.offset == other.offset,
            self.vartype == other.vartype
        ])

    @property
    def num_variables(self) -> int:
        """The number of variables in the BQM."""
        return len(self.linear)

    @property
    def num_interactions(self) -> int:
        """The number of nonzero interactions in the BQM."""
        return len(self.quadratic)

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the BQM, i.e the number of variables and number of edges"""
        return self.num_variables, self.num_interactions

    def set_offset(self, offset: Bias) -> None:
        """Set the offset of the BQM."""
        self.offset = offset

    def set_variable(self, v: Variable, bias: Bias, vartype: Vartype|None = None) -> None:
        """Set a variable of the BQM.
        If the variable already exists, overwrite its bias, else, create a new variable.
        """
        if vartype is not None and vartype is not self.vartype:
            if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
                bias *= -2
            elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
                bias *= -1/2
            else:
                raise ValueError(f'Vartype unknown: {Vartype}')
        self.linear[v] = bias

    def set_variables_from(self, linear: LinearLike, vartype: Vartype|None = None):
        """Set variables of the BQM."""
        linear = convert_to_linear(linear)
        for v, bias in linear.items():
            self.set_variable(v, bias, vartype)

    def set_interaction(self, u: Variable, v: Variable, bias: Bias, vartype: Vartype|None = None):
        """Set an interaction of the BQM.
        Set/Overwrite the coupling term between variables u and v of the BQM.
        If variables u and/or v do not exists, create them first (with linear bias 0).
        """
        if u == v:
            raise ValueError(f'Self-coupling interactions such as ({u},{v}) are not allowed')
        for var in (u, v):
            if var not in self.linear:
                self.linear[var] = 0
        if vartype is not None and vartype is not self.vartype:
            if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
                self.linear[u] += 2 * bias
                self.linear[v] += 2 * bias
                self.offset += -bias
                bias *= -4
            elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
                self.linear[u] += -1/4 * bias
                self.linear[v] += -1/4 * bias
                self.offset += 1/4 * bias
                bias *= -1/4
            else:
                raise ValueError(f'Vartype unknown: {Vartype}')
        self.quadratic[frozenset({u, v})] = bias

    def set_interactions_from(self, quadratic: QuadraticLike, vartype: Vartype|None = None):
        """Set interactions of the BQM."""
        quadratic = convert_to_quadratic(quadratic)
        for (u, v), bias in quadratic.items():
            self.set_interaction(u, v, bias, vartype)

    def remove_variable(self, v: Variable):
        """Remove variable from the BQM.
        The corresponding linear term and all interactions involving the given variables are removed.
        """
        if v not in self.linear:
            return
        del self.linear[v]
        for e in list(self.quadratic.keys()):
            if v in e:
                del self.quadratic[e]

    def remove_interaction(self, e: Collection[Variable, Variable]):
        """Remove interaction from the BQM."""
        e = frozenset(e)
        if e in self.quadratic:
            del self.quadratic[e]

    def scale(self, scalar: Bias):
        """Scale all components of the BQM."""
        for v in self.linear:
            self.linear[v] *= scalar
        for e in self.quadratic:
            self.quadratic[e] *= scalar
        self.offset *= scalar

    def change_vartype(self, vartype: Vartype):
        """Change the vartype of the given BQM encoding in place."""
        if vartype is self.vartype:
            return
        if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
            self.linear, self.quadratic, self.offset = self.binary_to_spin(self.linear, self.quadratic, self.offset)
        elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
            self.linear, self.quadratic, self.offset = self.spin_to_binary(self.linear, self.quadratic, self.offset)
        else:
            raise ValueError(f'Vartype unknown: {Vartype}')

    @staticmethod
    def spin_to_binary(linear: dict[Variable, Bias], quadratic: dict[frozenset[Variable, Variable], Bias], offset: Bias):
        """Calculate spin-to-binary conversion.
        Static method to convert a given linear, quadratic and offset from spin encoding to their binary counterpart.
        """
        linear = { v : 1/4 * sum([ bias for (e, bias) in quadratic.items() if v in e ]) - 1/2 * bias for (v, bias) in linear.items() }
        quadratic = { e : -1/4 * bias for (e, bias) in quadratic.items() }
        offset = 1/4 * sum(quadratic.values()) + 1/2 * sum(linear.values())
        return linear, quadratic, offset

    @staticmethod
    def binary_to_spin(linear: dict[Variable, Bias], quadratic: dict[frozenset[Variable, Variable], Bias], offset: Bias):
        """Calculate binary-to-spin conversion.
        Static method to convert a given linear, quadratic and offset from binary encoding to their spin counterpart.
        """
        linear = { v : 2 * sum([ bias for (e, bias) in quadratic.items() if v in e ]) - 2 * bias for (v, bias) in linear.items() }
        quadratic = { e : -4 * bias for (e, bias) in quadratic.items() }
        offset = - sum(quadratic.values()) + sum(linear.values())
        return linear, quadratic, offset

    def copy(self) -> BinaryQuadraticModel:
        """Create hard-copy of the BQM object"""
        return BinaryQuadraticModel(self.linear, self.quadratic, self.offset, self.vartype)

    def to_qubo(self, variable_order: Sequence[Variable]|None = None) -> tuple[np.ndarray, Bias]:
        """Extract a QUBO matrix for this BQM.
        Variable_order may be supplied to fix the order of variables in the matrix.
        Note: Q is an upper triangular matrix
        """
        self.change_vartype(Vartype.BINARY)
        Q = np.zeros((self.num_variables)*2, dtype=float)
        if variable_order is None:
            idx = { v : i for i, v in enumerate(self.linear) } # essentially random order
        else:
            idx = { v : i for i, v in enumerate(variable_order) }
        try:
            for v, bias in self.linear:
                Q[idx[v], idx[v]] = bias
            for (u, v), bias in self.quadratic:
                iu, iv = idx[u], idx[v]
                if iu < iv:
                    Q[iu, iv] = bias
                else:
                    Q[iv, iu] = bias
        except KeyError:
            raise ValueError(f'variable {v} missing from variable_order')
        return Q, self.offset

    @classmethod
    def from_qubo(cls, Q: np.ndarray, offset: Bias = 0.0, variable_order: Sequence[Variable]|None = None) -> BinaryQuadraticModel:
        """Create BQM from a QUBO matrix.
        Variable_order may be supplied to label the nodes of the BQM.
        If not supplied, integers (starting at 0) will be used as variable names.
        """
        if Q.ndim != 2:
            raise ValueError('Given QUBO matrix is not 2-dimensional')
        if Q.shape[0] != Q.shape[1]:
            raise ValueError('Given QUBO matrix is not square')
        if variable_order is None:
            variable_order = list(range(Q.shape[0]))
        try:
            bqm = cls({}, {}, offset, Vartype.BINARY)
            it = np.nditer(Q, flags=['multi_index'])
            for bias in it:
                row, col = it.multi_index
                if row == col:
                    bqm.set_variable(variable_order[row], bias)
                elif row > col:
                    bqm.set_interaction(variable_order[row], variable_order[col], bias)
                else:
                    continue
        except IndexError:
            raise ValueError('Given variable_order should have the same size as the QUBO matrix')
        return bqm

    def to_ising(self, variable_order: Sequence[Variable]|None = None) -> tuple[np.ndarray, np.ndarray[Bias]]:
        """Extract Ising matrix/vector representation for this BQM.
        Variable_order may be supplied to fix the order of variables in the matrix.
        Note: J is an upper triangular matrix.
        """
        self.change_vartype(Vartype.SPIN)
        h = np.zeros((self.num_variables), dtype=float)
        J = np.zeros((self.num_variables)*2, dtype=float)
        if variable_order is None:
            idx = { v : i for i, v in enumerate(self.linear) } # essentially random order
        else:
            idx = { v : i for i, v in enumerate(variable_order) }
        try:
            for v, bias in self.linear:
                h[idx[v]] = bias
            for (u, v), bias in self.quadratic:
                iu, iv = idx[u], idx[v]
                if iu < iv:
                    J[iu, iv] = bias
                else:
                    J[iv, iu] = bias
        except KeyError:
            raise ValueError(f'variable {v} missing from variable_order')
        return h, J

    @classmethod
    def from_ising(cls, h: np.ndarray, J: np.ndarray, offset: Bias = 0.0, variable_order: Sequence[Variable]|None = None) -> BinaryQuadraticModel:
        """Create BQM from Ising matrix/vector represenation.
        Variable_order may be supplied to label the nodes of the BQM.
        If not supplied, integers (starting at 0) will be used as variable names.
        """
        if h.ndim != 1:
            raise ValueError('h must be a 1-dimensional ndarray')
        if J.ndim != 2:
            raise ValueError('J must be a 2-dimensional ndarray')
        if J.shape[0] != J.shape[1]:
            raise ValueError('J must be square')
        if J.shape[0] != h.size:
            raise ValueError('J and h must have matching sizes')
        if variable_order is None:
            variable_order = list(range(h.size))
        try:
            bqm = cls({}, {}, offset, Vartype.SPIN)
            for i, bias in np.ndenumerate(h):
                    bqm.set_variable(variable_order[i], bias)
            it = np.nditer(J, flags=['multi_index'])
            for bias in it:
                i, j = it.multi_index
                if i > j:
                    bqm.set_interaction(variable_order[i], variable_order[j], bias)
                else:
                    continue
        except IndexError:
            raise ValueError('Given variable_order does not contain all nodes')
        return bqm

    def to_file(self, file: Path):
        raise NotImplementedError()

    @classmethod
    def from_file(cls, file: Path):
        raise NotImplementedError()
