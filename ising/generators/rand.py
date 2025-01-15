import random
import numpy as np
from collections.abc import Sequence

from ising.model import IsingModel

def uniform(
        adj: np.ndarray,
        linear: np.ndarray | None = None,
        low: float = 0,
        high: float = 1,
        seed: int | None = None) -> IsingModel:
    random.seed(seed)
    def gen():
        while True:
            yield random.uniform(low, high)
    return IsingModel.from_adjacency(adj, linear, gen())


def randint(
        adj: np.ndarray,
        linear: np.ndarray | None = None,
        low: int = 0,
        high: int = 1,
        seed: int | None = None) -> IsingModel:
    random.seed(seed)
    def gen():
        while True:
            yield random.randint(low, high)
    return IsingModel.from_adjacency(adj, linear, gen())


def sample(
        adj: np.ndarray,
        linear: np.ndarray | None = None,
        population: Sequence = [0, 1],
        counts=None,
        seed: int | None = None) -> IsingModel:
    random.seed(seed)
    def gen():
        while True:
            yield random.sample(population, counts=counts, k=1)[0]
    return IsingModel.from_adjacency(adj, linear, gen())
