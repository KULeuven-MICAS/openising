import pathlib
import numpy as np
import scipy.sparse.linalg as spalg

from ising.utils.numpy import triu_to_symm
from ising.stages.model.ising import IsingModel

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


def return_c0(model: IsingModel) -> float:
    """Returns the optimal c0 value for simulated bifurcation.

    Args:
        model (IsingModel): the Ising model that will be solved with simulated Bifurcationl.

    Returns:
        float: the c0 hyperaparameter.
    """
    return 0.5 / (
        np.sqrt(model.num_variables)
        * np.sqrt(np.sum(np.power(model.J, 2)) / (model.num_variables * (model.num_variables - 1)))
    )


def return_G(J: np.ndarray) -> float:
    """Returns the optimal latch resistant value for the given problem.

    Args:
        J (np.ndarray): the coefficient matrix of the problem that will be solved with BRIM.

    Returns:
        float: the latch resistance.
    """
    sumJ = np.sum(np.abs(triu_to_symm(J)), axis=0)
    return np.average(sumJ) * 2


def return_q(problem: IsingModel) -> float:
    """Returns the optimal value for the penalty parameter q for the SCA solver.

    Args:
        problem (IsingModel): the problem that will be solved with SCA.

    Returns:
        float: the penalty parameter q.
    """
    eig = np.abs(spalg.eigs(triu_to_symm(-problem.J), 1)[0][0])
    return eig / 2
