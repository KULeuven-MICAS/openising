import numpy as np
import time

from ising.stages.model.ising import IsingModel

def MIMO_to_Ising(
    H: np.ndarray, x: np.ndarray, SNR: float, Nr: int, Nt: int, M: int, seed:int=0
) -> tuple[IsingModel, np.ndarray]:
    """Transforms the MIMO model into an Ising model.

    Args:
        H (np.ndarray): The transfer function matrix.
        x (np.ndarray): the input signal.
        T (np.ndarray): the transformation matrix to transform the input signal to Ising format.
        SNR (float): the signal to noise ratio.
        Nr (int): the amount of input signals.
        Nt (int): the amount of output signals.
        M (int): the considered QAM scheme.
        seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
        model,xtilde,ytilde (tuple[IsingModel, np.ndarray]): the generated Ising model and transformed input
                                                             and output signals.
    """
    if np.linalg.norm(np.imag(x)) == 0:
        # BPSK scheme
        r = 1
        Nx = np.shape(x)[0]
        Ny = 2*Nx
    else:
        r = int(np.ceil(np.log2(np.sqrt(M))))
        Nx = np.shape(x)[0]*2
        Ny = Nx

    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    # Compute the amplitude of the noise
    power_x = (np.abs(x)**2)
    SNR = 10 ** (SNR / 10)
    var_noise = np.sqrt(power_x / SNR)
    n = var_noise*(np.random.randn(Nr) + 1j * np.random.randn(Nr)) / (np.sqrt(2))

    # Compute the received symbols
    y = H @ x + n
    ytilde = np.block([np.real(y), np.imag(y)])

    Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])

    T = np.block([2**(r-i)*np.eye(Ny, Nx) for i in range(1, r+1)])

    xtilde = np.block([np.real(x), np.imag(x)])

    ones_end = np.eye(Ny, Nx) @ np.ones((Nx,))
    constant = ytilde.T@ytilde - 2*ytilde.T @ Htilde @ (T@np.ones((r*Nx,)) - \
                                        (np.sqrt(M)-1)*ones_end)

    bias = 2*(ytilde - Htilde@(T@np.ones((r*Nx,))-(np.sqrt(M)-1)*ones_end))
    bias = bias.T @ Htilde @ T
    coupling = -2*T.T @ Htilde.T @ Htilde @ T
    diagonal = np.diag(coupling)
    constant -= np.sum(diagonal)/2

    coupling = np.triu(coupling, k=1)
    return IsingModel(coupling, bias, constant, name=f"MIMO_{SNR}"), xtilde, ytilde


def compute_difference(sigma_optim: np.ndarray, x: np.ndarray, M:int) -> float:
    """Computes the symbol error rate between the optimal solution and the computed solution.

    Args:
        sigma_optim (np.ndarray): the optimal solution.
        x (np.ndarray): the computed solution.
        M (int): the modulation scheme.
    Returns:
        BER (float): the bit error rate between the two solutions.
    """
    r = int(np.ceil(np.log2(np.sqrt(M))))

    N = np.shape(x)[0]
    nb_runs = np.shape(x)[1]

    # Compute the calculated symbols
    T = np.block([[2 ** (r - i) * np.eye(N) for i in range(1, r + 1)]])
    x_optim = T @ (sigma_optim + np.ones((r * N,nb_runs))) - (np.sqrt(M) - 1) * np.ones((N,nb_runs))
    BER = np.sum(np.abs(x - x_optim)/2, axis=0)/(np.sqrt(M)*N)
    return BER
