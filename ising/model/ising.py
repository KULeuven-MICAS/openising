from __future__ import annotations
import pathlib
from collections.abc import Callable, Iterable
import numpy as np
import h5py

import ising.utils.numpy as npu

Bias = int | float


class IsingModel:
    """Encodes a binary quadratic model."""

    __slots__ = ["J", "h", "c"]

    def __init__(self, J: np.ndarray, h: np.ndarray, c: Bias = 0):
        if not isinstance(h, np.ndarray) or not h.ndim == 1:
            raise ValueError("h must be a vector")
        if not isinstance(J, np.ndarray) or not npu.is_square(J):
            raise ValueError("J must be a square matrix")
        if not npu.is_triu(J, k=1):
            raise ValueError("J must be a strictly upper triangular matrix")
        if not len(h) == J.shape[0]:
            raise ValueError(f"h ({h.shape}) and J ({J.shape}) are not compatible")
        self.J = J
        self.h = h
        self.c = c

    def __repr__(self) -> str:
        return f"IsingModel(\n J={str(self.J).replace('\n ', '\n    ')},\n h={self.h},\n c={self.c}\n)"

    def __len__(self) -> int:
        return self.num_variables

    @property
    def num_variables(self) -> int:
        """
        The number of variables in the BQM.

        :return N (int): the number of variables (nodes)
        """
        return len(self.h)

    @property
    def num_interactions(self) -> int:
        """
        The number of nonzero interactions in the BQM.

        :return |J| (int): the size of the set of edges
        """
        return np.count_nonzero(self.J)

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the BQM, i.e the number of variables and number of edges.

        :return (N, |J|) (tuple[int, int]): the shape of the model
        """
        return self.num_variables, self.num_interactions

    def copy(self) -> IsingModel:
        """
        Create hard-copy of the BQM object.

        :return model (IsingModel): hard-copy of the original object
        """
        return IsingModel(self.J, self.h)

    def evaluate(self, sample: np.ndarray) -> Bias:
        """
        Compute the Hamiltonian given a sample.

        :param np.ndarray sample: vector of spin values (1 or -1)
        :return H (float): value of the Hamiltonian
        """
        return -np.dot(sample.T, np.dot(self.J, sample)) - np.dot(self.h.T, sample) + self.c

    @classmethod
    def from_qubo(cls, Q: np.ndarray) -> IsingModel:
        if not isinstance(Q, np.ndarray) or not npu.is_square(Q) or not npu.is_triu(Q):
            raise ValueError("Q must be a square upper triangular matrix")
        J = -(1 / 4) * Q.copy()
        h = -(1 / 2) * Q.diagonal().copy() - (1 / 4) * np.sum(npu.triu_to_symm(Q), axis=1)
        c = (1 / 4) * np.sum(Q) + (1 / 4) * np.sum(Q.diagonal())
        return cls(J, h, c)

    def to_qubo(self) -> tuple[np.ndarray, Bias]:
        Q = (-4) * self.J
        Q.diagonal()[:] = 2 * (np.sum(npu.triu_to_symm(self.J), axis=1) + self.h)
        c = -np.sum(self.J) + np.sum(self.h)
        return Q, c

    @classmethod
    def from_file(cls, file: pathlib.Path):
        with h5py.File(file) as f:
            J_dset = f.get("J")
            h = f.get("h")
            c = f.get("c", 0)
            size = f.attrs.get("size", h.size)
            J = np.zeros((size, size), dtype=J_dset.dtype)
            J[np.triu_indices(size, k=1)] = J_dset
            return cls(J, h, c)

    def to_file(self, file: pathlib.Path):
        with h5py.File(file) as f:
            f.create_dataset("J", data=self.J[np.triu_indices(self.num_variables, k=1)])
            f.create_dataset("h", data=self.h)
            f.create_dataset("c", data=self.c)
            f.attrs["size"] = self.num_variables

    @classmethod
    def from_adjacency(
            cls,
            adj: np.ndarray,
            linear: np.ndarray | None = None,
            bias_generator: Bias | Callable | Iterable = 1) -> IsingModel:
        if isinstance(bias_generator, Bias):
            f = lambda: bias_generator
        elif isinstance(bias_generator, Callable):
            f = bias_generator
        elif isinstance(bias_generator, Iterable):
            iterator = iter(bias_generator)
            f = lambda: next(iterator)
        else:
            raise ValueError("bias_generator is neither a valid numpy scalar, nor a Callable, nor a Generator")

        adj = adj.astype(bool)
        adj[np.tril_indices_from(adj)] = False

        J = np.zeros_like(adj, dtype=float)
        J[adj] = np.array([f() for _ in range(np.sum(adj))])

        if linear is None:
            h = np.zeros(adj.shape[0], dtype=float)
        else:
            linear = linear.astype(bool)
            h = np.zeros_like(linear, dtype=float)
            h[linear] = np.array([f() for _ in range(np.sum(linear))])

        return cls(J, h)
