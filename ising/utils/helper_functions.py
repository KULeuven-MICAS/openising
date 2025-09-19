import pathlib

def make_directory(path: pathlib.Path) -> None:
    """Makes the given directory if it does not exist.

    Args:
        path (pathlib.Path): the directory to create
    """
    path.mkdir(parents=True, exist_ok=True)

def return_rx(num_iter: int, r_init: float, r_final: float) -> float:
    """Returns the change rate of SA/SCA hyperparameters

    Args:
        num_iter (int): amount of iterations.
        r_init (float): the initial value of the hyperparameter.
        r_final (float): the end value of the hyperparameter.

    Returns:
        float: the change rate of the hyperarameter.
    """
    return (r_final / r_init) ** (1 / (num_iter + 1))
# def return_G(J: np.ndarray) -> float:
#     """Returns the optimal latch resistant value for the given problem.

#     Args:
#         J (np.ndarray): the coefficient matrix of the problem that will be solved with BRIM.

#     Returns:
#         float: the latch resistance.
#     """
#     sumJ = np.sum(np.abs(triu_to_symm(J)), axis=0)
#     return np.average(sumJ) * 2
