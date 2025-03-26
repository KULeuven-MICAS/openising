import pathlib
import numpy as np
import scipy.sparse.linalg as spalg

from ising.utils.HDF5Logger import return_metadata

from ising.model.ising import IsingModel
from ising.solvers.BRIM import BRIM
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.SA import SASolver
from ising.solvers.DSA import DSASolver
from ising.solvers.Multiplicative import Multiplicative

from ising.utils.numpy import triu_to_symm


def make_directory(path: pathlib.Path) -> None:
    """Makes the given directory if it does not exist.

    Args:
        path (pathlib.Path): the directory to create
    """
    path.mkdir(parents=True, exist_ok=True)


def parse_hyperparameters(args: dict, num_iter: int) -> dict[str:]:
    """Parses the arguments needed for the solvers.

    Args:
        args (dict): the command line arguments.
        num_iter (int): amount of iterations

    Returns:
        dict[str:Any]: the hyperparameters for the solvers.
    """
    hyperparameters = dict()

    # Multiplicative parameters
    hyperparameters["dtMult"] = float(args.dtMult)
    hyperparameters["coupling_annealing"] = bool(int(args.coupling_annealing))

    # BRIM parameters
    dtBRIM = float(args.dtBRIM)
    hyperparameters["dtBRIM"] = dtBRIM
    hyperparameters["C"] = float(args.C)
    hyperparameters["stop_criterion"] = float(args.stop_criterion)
    hyperparameters["initial_temp_cont"] = float(args.T_cont)
    hyperparameters["end_temp_cont"] = float(args.T_final_cont)

    # SA parameters
    hyperparameters["initial_temp"] = float(args.T)
    Tfin = float(args.T_final)
    hyperparameters["cooling_rate"] = (
        return_rx(num_iter, hyperparameters["initial_temp"], Tfin) if hyperparameters["initial_temp"] != 0 else 0.0
    )
    hyperparameters["seed"] = int(args.seed)

    # SCA parameters
    hyperparameters["q"] = float(args.q)

    # SB parameters
    hyperparameters["dtSB"] = float(args.dtSB)
    hyperparameters["a0"] = float(args.a0)
    hyperparameters["c0"] = float(args.c0)

    return hyperparameters


def get_best_found_gurobi(gurobi_files: list[pathlib.Path]) -> list[float]:
    """Returns a list of the best found energies in the gurobi files.

    Args:
        gurobi_files (list[pathlib.Path]): the gurobi files.

    Returns:
        list[float]: list of the best found energies.
    """
    best_found_list = []
    for file in gurobi_files:
        best_found = return_metadata(file, "solution_energy")
        best_found_list.append(best_found)
    return best_found_list


def run_solver(
    solver: str,
    num_iter: int,
    s_init: np.ndarray,
    model: IsingModel,
    logfile: pathlib.Path | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, float]:
    """Solves the given problem with the specified solver.

    Args:
        solver (str): The solver
        num_iter (int): amount of iterations
        s_init (np.ndarray): initial state
        model (IsingModel): model to use
        logfile (pathlib.Path | None, optional): path to logfile to store data. Defaults to None.

    Returns:
        optim_state,optim_energy (tuple[np.ndarray, float]): optimal state and energy of the specified solver.
    """
    optim_state = np.zeros((model.num_variables,))
    optim_energy = None
    solvers = {
        "BRIM": (
            BRIM().solve,
            ["dtBRIM", "C", "stop_criterion", "initial_temp_cont", "end_temp_cont", "seed", "coupling_annealing"],
        ),
        "Multiplicative": (Multiplicative().solve,
                           ["dtMult", "initial_temp_cont", "end_temp_cont", "seed", "coupling_annealing"]),
        "SA": (SASolver().solve, ["initial_temp", "cooling_rate", "seed"]),
        "DSA": (DSASolver().solve, ["initial_temp", "cooling_rate", "seed"]),
        "SCA": (SCA().solve, ["initial_temp", "cooling_rate", "q", "r_q", "seed"]),
        "bSB": (ballisticSB().solve, ["c0", "dtSB", "a0"]),
        "dSB": (discreteSB().solve, ["c0", "dtSB", "a0"]),
    }
    if solver in solvers:
        func, params = solvers[solver]
        chosen_hyperparameters = {key: hyperparameters[key] for key in params if key in hyperparameters}
        optim_state, optim_energy = func(
            model=model,
            initial_state=s_init,
            num_iterations=num_iter,
            file=logfile,
            **chosen_hyperparameters,
        )
    return optim_state, optim_energy


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


def compute_list_from_arg(arg: str, step: int = 1) -> np.ndarray:
    """Returns a list of integers given a argument string and step size.

    Args:
        arg (str): the argument holding the range information.
        step (int, optional): the step size. Defaults to 1.

    Returns:
        np.ndarray: the list of integers.
    """
    arg_list = arg.split()
    return np.array(range(int(arg_list[0]), int(arg_list[1]) + 1, step))
