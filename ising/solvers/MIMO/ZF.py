import numpy as np

from ising.solvers.base import SolverBase


class ZF(SolverBase):
    def __init__(self):
        self.name= "ZF_decoder"


    def solve(self, H_channel: np.ndarray, y_vector:np.ndarray, M:int):
        H_inv = np.linalg.pinv(H_channel)

        x_hat = H_inv @ y_vector

        if M == 2:
            # BPSK scheme
            symbols = np.array([-1.0, 1.0])
        else:
            L = int(round(np.sqrt(M)))
            symbols = np.arange(-(L - 1), L, 2, dtype=float)

        symbols_repeated = np.repeat(symbols.reshape(-1,1), x_hat.shape[0], axis=1).T
        for i in range(x_hat.shape[1]):
            x_real = np.real(x_hat[:, i])
            x_imag = np.imag(x_hat[:, i])
            idx_real = np.argmin(np.abs(x_real.reshape(-1,1) - symbols_repeated), axis=1).reshape((-1,))
            idx_imag = np.argmin(np.abs(x_imag.reshape(-1,1) - symbols_repeated), axis=1).reshape((-1,))
            x_hat[:, i] = symbols[idx_real] + symbols[idx_imag]*1j

        return x_hat
