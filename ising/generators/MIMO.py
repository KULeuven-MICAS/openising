import numpy as np
import time

from ising.model.ising import IsingModel


def MU_MIMO(Nt: int, Nr: int, M: int, seed: int = 1) -> tuple[IsingModel, np.ndarray]:
    """Generates a MU-MIMO model using section IV-A of [this paper](https://arxiv.org/pdf/2002.02750).
    This is consecutively transformed into an Ising model.

    Args:
        Nt (int): The amount of users.
        Nr (int): The amount of antennas at the Base Station.
        M (int): the considered QAM scheme.
        seed (int, optional): The seed for the random number generator. Defaults to 1.

    Returns:
        tuple[IsingModel, np.ndarray]: the generated Ising model and the solution.
    """
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    r = int(np.ceil(np.log2(np.sqrt(M))))
    symbols = np.concatenate(
        ([-np.sqrt(M) + i for i in range(1, 1 + 2 * r, 2)], [np.sqrt(M) - i for i in range(1, 1 + 2 * r, 2)])
    )

    phi_u     = 120 * (np.random.random((10, Nt)) - 0.5)
    phi_u.sort()
    mean_phi  = np.mean(phi_u, axis=0)
    sigma_phi = np.random.normal(0, 1, (Nt,))

    H = np.zeros((Nr, Nt), dtype='complex_')
    for i in range(Nt):
        C     = np.zeros((Nr, Nr), dtype="complex_")
        phi   = mean_phi[i]
        sigma = sigma_phi[i]
        for m in range(Nr):
            for n in range(Nr):
                d = spacing_BS_antennas(m, n)
                C[m, n] = np.exp(2*np.pi*1j*d*np.sin(phi))* np.exp(
                    -(sigma**2) / 2 * (2 * np.pi * d * np.cos(phi)) ** 2
                )
        D, V = np.linalg.eig(C)
        hu = V @ np.diag(D)**0.5 @ V.conj().T @ (np.random.normal(0, 1, (Nr,)) + 1j*np.random.normal(0, 1, (Nr,)))
        H[:, i] = hu

    return H, symbols


def spacing_BS_antennas(m, n):
    return np.abs(m - n)


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
        tuple[IsingModel, np.ndarray]: the generated Ising model and transformed input signal.
    """
    r = int(np.ceil(np.log2(np.sqrt(M))))

    Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])

    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    # Compute the amplitude of the noise
    power_x = np.mean(np.abs(x)**2)
    SNR = 10 ** (SNR / 10)
    var_noise = np.sqrt(power_x / SNR)
    n = var_noise*(np.random.randn(Nr) + 1j * np.random.randn(Nr)) / (np.sqrt(2))

    # Compute the received symbols
    y = H @ x + n
    ytilde = np.block([np.real(y), np.imag(y)])

    N = 2 * Nt
    T = np.block([[2 ** (r - i) * np.eye(N) for i in range(1, r + 1)]])
    xtilde = np.block([np.real(x), np.imag(x)])
    z = ytilde - (Htilde @ (T @ np.ones(r * N))) + ((np.sqrt(M) - 1) * Htilde @ np.ones(N))

    # Set up the Ising model
    J = -T.T @ Htilde.T @ Htilde @ T
    J = np.triu(J, k=1)
    h = np.transpose(2 * z.T @ Htilde @ T)
    c = np.inner(z, z)

    return IsingModel(J, h, c), xtilde


def compute_difference(sigma_optim: np.ndarray, x: np.ndarray, M:int) -> float:
    """Computes the symbol error rate between the optimal solution and the computed solution.

    Args:
        sigma_optim (np.ndarray): the optimal solution.
        x (np.ndarray): the computed solution.
        M (int): the modulation scheme.
    Returns:
        SER (float): the bit error rate between the two solutions.
    """
    r = int(np.ceil(np.log2(np.sqrt(M))))

    N = np.shape(x)[0]

    # Compute the calculated symbols
    T = np.block([[2 ** (r - i) * np.eye(N) for i in range(1, r + 1)]])
    x_optim = T @ (sigma_optim + np.ones((r * N,))) - (np.sqrt(M) - 1) * np.ones((N,))
    print("optimal solution: ", x_optim)
    BER = np.sum(np.abs(x - x_optim)/2)/(2*N)
    return BER
