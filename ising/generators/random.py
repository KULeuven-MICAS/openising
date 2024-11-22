import random as rd
import numpy as np

from ising.model.ising import IsingModel


def random(
        adj: np.ndarray,
        linear: np.ndarray | None = None,
        low: float = 0,
        high: float = 1,
        seed: int | None = None) -> IsingModel:
    rd.seed(seed)
    def gen():
        while True:
            yield rd.uniform(low, high)
    return IsingModel.from_adjacency(adj, linear, gen())


def randint(
        adj: np.ndarray,
        linear: np.ndarray | None = None,
        low: int = 0,
        high: int = 1,
        seed: int | None = None) -> IsingModel:
    rd.seed(seed)
    def gen():
        while True:
            yield rd.randint(low, high)
    return IsingModel.from_adjacency(adj, linear, gen())
