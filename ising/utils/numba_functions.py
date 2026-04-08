import numpy as np
from numba import njit


@njit(cache=True)
def evaluate_ising(sample: np.ndarray, J: np.ndarray, h: np.ndarray, c: float):
    acc = 0.0
    res = np.empty(np.shape(h), dtype=np.float32)
    res[:] = J @ sample


    acc = -sample.transpose() @ res - h.transpose() @ sample + c
    return acc
