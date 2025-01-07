import numpy as np

from ising.model.ising import IsingModel


def MU_MIMO(Nt: int, Nr: int, M: int, SNR: float, seed:int=1) -> tuple[IsingModel, np.ndarray]:
    """Generates a MU-MIMO model using section IV-A of [this paper](https://arxiv.org/pdf/2002.02750).
    This is consecutively transformed into an Ising model.

    Args:
        Nt (int): The amount of users.
        Nr (int): The amount of antennas at the Base Station.
        M (int): the considered QAM scheme.
        SNR (float): the Signal to Noise Ratio.

    Returns:
        tuple[IsingModel, np.ndarray]: the generated Ising model and the solution.
    """
    np.random.seed(seed)
    r = int(np.ceil(np.log2(np.sqrt(M))))
    symbols = np.concatenate(
        ([-np.sqrt(M) + i for i in range(1, 1 + 2 * r, 2)], [np.sqrt(M) - i for i in range(1, 1 + 2 * r, 2)])
    )

    H = np.random.normal(0, 1, (Nr, Nt)) + 1j*np.random.normal(0, 1, (Nr, Nt))

    return H, symbols


def MIMO_to_Ising(H, x, T, SNR, Nr, Nt, M):
    r = int(np.ceil(np.log2(np.sqrt(M))))

    Htilde = np.block([[np.real(H), -np.imag(H)],
                    [np.imag(H), np.real(H)]])

    amp = np.average(np.abs(x))/10**(SNR/20)
    n = 1/amp*(np.random.normal(0, 1, (Nr,)) + 1j*np.random.normal(0, 1, (Nr,)))

    y = H @ x + n
    ytilde = np.block([np.real(y), np.imag(y)])
    N = 2*Nt
    T = np.block([[2**(r-i)*np.eye(N) for i in range(1, r+1)]])
    xtilde = np.block([np.real(x), np.imag(x)])
    z = ytilde - (Htilde @ (T@np.ones(r*N))) + ((np.sqrt(M) -1)*Htilde@np.ones(N))

    J = -T.T@Htilde.T@Htilde@T
    J = np.triu(J, k=1)
    h = np.transpose(2*z.T@Htilde@T)
    c = np.inner(z, z)

    return IsingModel(J, h, c), xtilde


def compute_difference(sigma_optim:np.ndarray, x:np.ndarray, M):
    """Computes the relative error between the optimal solution and the computed solution.

    Args:
        sigma_optim (np.ndarray): the optimal solution.
        x (np.ndarray): the computed solution.

    Returns:
        float: the difference between the two solutions.
    """
    r = int(np.ceil(np.log2(np.sqrt(M))))

    N = np.shape(x)[0]
    T = np.block([[2**(r-i)*np.eye(N) for i in range(1, r+1)]])
    x_optim = T @ (sigma_optim + np.ones((r*N,))) - (np.sqrt(M) - 1)*np.ones((N,))
    BER = np.count_nonzero(x_optim - x) / N
    return BER
