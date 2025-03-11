import numpy as np

from ising.model.ising import IsingModel

def dual_decomposition(model1: tuple[np.ndarray, IsingModel], model2: tuple[np.ndarray, IsingModel],
                       A: np.ndarray, C:np.ndarray) -> tuple[np.ndarray, IsingModel]:
    pass