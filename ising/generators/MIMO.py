import numpy as np



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
