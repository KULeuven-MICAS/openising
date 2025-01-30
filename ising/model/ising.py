from __future__ import annotations
import pathlib
from collections.abc import Callable, Iterable
import numpy as np
import h5py

import ising.utils.numpy as npu


class IsingModel:
    """
    A class representing an Ising model.

    Attributes:
        J (np.ndarray): A square strictly upper triangular matrix representing interactions between variables.
        h (np.ndarray): A vector of bias values for each variable.
        c (float): A constant term in the Hamiltonian.
    """

    def __init__(self, J: np.ndarray, h: np.ndarray, c: float = 0) -> None:
        """
        Initialize an Ising model with the specified interaction matrix, bias vector, and constant.

        Args:
            J (np.ndarray): The interaction matrix (square strictly upper triangular matrix).
            h (np.ndarray): The bias vector.
            c (float): The constant term (default is 0).
        """
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
        self.transformation_history = []

    def __repr__(self) -> str:
        return f"IsingModel(\n J={str(self.J).replace('\n ', '\n    ')},\n h={self.h},\n c={self.c}\n)"

    def __len__(self) -> int:
        return self.num_variables

    @property
    def num_variables(self) -> int:
        """
        The number of variables in the Ising model.

        Returns:
            int: the number of variables (nodes).
        """
        return len(self.h)

    @property
    def num_interactions(self) -> int:
        """
        The number of nonzero interactions in the Ising model.

        Returns:
            int: the size of the set of edges.
        """
        return np.count_nonzero(self.J)

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the Ising model, i.e the number of variables and number of edges.

        Returns:
            typle[int, int]: A tuple representing the shape of the model (N, |J|).
        """
        return self.num_variables, self.num_interactions

    @property
    def mean(self) -> float:
        """
        The mean of biases in the Ising model.

        Returns:
            float: the mean value of all linear and quadratic biases.
        """
        biases = np.concatenate([self.J[np.triu_indices(self.num_variables, k=1)], self.h])
        return np.mean(biases)

    @property
    def variance(self) -> float:
        """
        The variance of biases in the Ising model.

        Returns:
            float: the variance value of all linear and quadratic biases.
        """
        biases = np.concatenate([self.J[np.triu_indices(self.num_variables, k=1)], self.h])
        return np.var(biases)

    def copy(self) -> IsingModel:
        """
        Create hard copy of the IsingModel object.

        Returns:
            IsingModel: A new IsingModel instance with the same J, h, and c values.
        """
        return IsingModel(self.J, self.h)

    def translate(self, displacement: float, track: bool = True) -> None:
        """
        Translate the Ising model by a given displacement value.

        Args:
            displacement (float): The value to add to both the J and h matrices, as well as the constant term.
        """
        triu = np.triu_indices(self.num_variables, k=1)
        self.J[triu] = np.add(self.J[triu], displacement, casting="safe")
        self.h = np.add(self.h, displacement, casting="safe")
        self.c = np.add(self.c, displacement, casting="safe")
        if track:
            self.transformation_history.append(("translate", displacement))

    def scale(self, factor: float, track: bool = True) -> None:
        """
        Scale the Ising model by a given factor.

        Args:
            factor (float): The scaling factor to apply to J, h, and c.
        """
        triu = np.triu_indices(self.num_variables, k=1)
        self.J[triu] = np.multiply(self.J[triu], factor, casting="safe")
        self.h = np.multiply(self.h, factor, casting="safe")
        self.c = np.multiply(self.c, factor, casting="safe")
        if track:
            self.transformation_history.append(("scale", factor))

    def normalize(self, mean: float = 0, variance: float = 1) -> None:
        """
        Normalize the Ising model such that the biases have the given mean and variance.

        Args:
            mean (float, optional): The desired mean for the normalized values (default is 0)
            variance (float, optional): The desired variance for the normalized values (default is 1)
        """
        displacement = mean - self.mean
        factor = np.sqrt(variance/self.variance) if self.variance != 0 else np.sqrt(variance)
        self.translate(displacement)
        self.scale(factor)

    def reconstruct(self, steps: int|None = None) -> None:
        """
        Reconstruct the Ising model, reversing the transformation history one-by-one.

        Args:
            steps (int, optional): The number of transformation steps that should be reversed,
                                   all if None (default is None).
        """
        if steps is None:
            steps = len(self.transformation_history)
        for _ in range(steps):
            if len(self.transformation_history) == 0:
                break
            transformation = self.transformation_history.pop()
            if transformation[0] == "translate":
                self.translate(-transformation[1], track=False)
            if transformation[0] == "scale":
                self.scale(1/transformation[1], track=False)


    def transform_to_no_h(self) -> IsingModel:
        """Generates a new Isingmodel with the bias incorporated into the coefficient matrix J.

        Returns:
            IsingModel: the new Ising Model.
        """
        new_h = self.h.reshape((-1, 1))
        new_J = np.block([[self.J, new_h],[np.zeros_like(new_h.T), 0]])
        return IsingModel(new_J, np.zeros(self.num_variables + 1), c=self.c)


    def evaluate(self, sample: np.ndarray) -> float:
        """
        Compute the Hamiltonian given a sample of spin values.

        Args:
            sample (np.ndarray): A vector of spin values (1 or -1)

        Returns:
            float: The calculated Hamiltonian value for the given sample.
        """
        return -np.dot(sample.T, np.dot(self.J, sample)) - np.dot(self.h.T, sample) + self.c

    @classmethod
    def from_qubo(cls, Q: np.ndarray) -> IsingModel:
        """
        Create an IsingModel from a QUBO matrix.

        Args:
            Q (np.ndarray): A square upper triangular matrix representing the QUBO problem.

        Returns:
            IsingModel: The corresponding IsingModel instance.
        """
        if not isinstance(Q, np.ndarray) or not npu.is_square(Q) or not npu.is_triu(Q):
            raise ValueError("Q must be a square upper triangular matrix")
        J = -(1 / 4) * np.triu(Q, k=1).copy()
        h = -(1 / 2) * Q.diagonal().copy() - (1 / 4) * np.sum(npu.triu_to_symm(Q), axis=1)
        c = (1 / 4) * np.sum(Q) + (1 / 4) * np.sum(Q.diagonal())
        return cls(J, h, c)

    def to_qubo(self) -> tuple[np.ndarray, float]:
        """
        Convert the IsingModel to a QUBO matrix representation.

        Returns:
            tuple[np.ndarray, float]: The QUBO matrix and the constant term c.
        """
        new_J = npu.triu_to_symm(self.J)
        Q = (-2) * new_J
        np.fill_diagonal(Q, 2 * (np.sum(new_J, axis=1) - self.h))
        c = -np.sum(new_J)/2 + np.sum(self.h) + self.c
        return Q, c

    @classmethod
    def from_file(cls, file: pathlib.Path) -> IsingModel:
        """
        Load an IsingModel from an HDF5 file.

        Args:
            file (pathlib.Path): The path to the HDF5 file.

        Returns:
            IsingModel: The loaded IsingModel instance.
        """
        with h5py.File(file) as f:
            J_dset = f.get("J")
            h = f.get("h")
            c = f.get("c", 0)
            size = f.attrs.get("size", h.size)
            J = np.zeros((size, size), dtype=J_dset.dtype)
            J[np.triu_indices(size, k=1)] = J_dset
            return cls(J, h, c)

    def to_file(self, file: pathlib.Path) -> None:
        """
        Save the IsingModel to an HDF5 file.

        Args:
            file (pathlib.Path): The path to the HDF5 file.

        """
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
            bias_generator: float | Callable | Iterable = 1) -> IsingModel:
        """
        Create an IsingModel from an adjacency matrix and linear bias vector by
        filling the biases with values taken from a bias_generator.

        Args:
            adj (np.ndarray): The adjacency matrix (transformable to bool-dtype).
            linear (np.ndarray, optional): A vector denoting the presence of linear biases (default is None).
            bias_generator (float | Callable | Iterable, optional): A value or generator function to sample biases from.

        Returns:
            IsingModel: The corresponding IsingModel instance.
        """
        if isinstance(bias_generator, (float, int)):
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
