import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))
figtop = TOP / "ising/flow/MIMO/connectivity_figs"
N = 2
M = 16
SNR_list = np.linspace(0, 10, 11)

for SNR in SNR_list:
    H, symbols = MU_MIMO(N, N, M, 0)
    x = np.random.choice(symbols, (N,)) + 1j*np.random.choice(symbols, (N,))
    model, xtilde = MIMO_to_Ising(H, x, SNR, N, N, M)

    newmodel = model.transform_to_no_h()
    print(f"{model.h=}")
    J = triu_to_symm(newmodel.J)

    plt.figure()
    plt.imshow(J, interpolation="nearest")
    plt.colorbar()
    plt.title(f"Connectivity matrix for MIMO model and {SNR=}")
    plt.savefig(figtop / f"connectivity_matrix_{SNR}.png")
    plt.show()

