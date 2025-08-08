import numpy as np
from numba import njit


@njit(cache=True)
def evaluate_ising(sample: np.ndarray, J: np.ndarray, h: np.ndarray, c: float):
    acc = 0.0
    res = np.empty(np.shape(h), dtype=np.float32)
    res[:] = J @ sample


    acc = -sample.transpose() @ res - h.transpose() @ sample + c
    return acc

@njit(cache=True)
def dvdt_solver(
    t: float,
    vt: np.ndarray,
    coupling: np.ndarray,
    bias: int,
    capacitance: float,
) -> np.ndarray:
    # set bias node to 1.
    if bias == 1:
        vt[-1] = 1.0

    # Compute the voltage change dv
    dv = np.zeros_like(vt, dtype=np.float32)
    dv[:] = coupling @ np.sign(vt)
    dv[:] = dv / capacitance

    # Clip voltages efficiently
    mask = ((dv > 0) & (vt >= 1)) | ((dv < 0) & (vt <= -1))
    dv[mask] = 0.0

    # if frozen_nodes is not None:
    #     for frozen_node in frozen_nodes:
    #         dv[frozen_node] = 0.0
    #     # dv[frozen_nodes] = 0.0

    # Ensure the bias node does not change
    if bias == 1:
        dv[-1] = 0.0
    return dv
