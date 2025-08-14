import pathlib
import numpy as np

from ising.utils.HDF5Logger import return_metadata

from ising.model.ising import IsingModel
from ising.solvers.BRIM import BRIM
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.SA import SASolver
from ising.solvers.DSA import DSASolver
from ising.solvers.Multiplicative import Multiplicative
from ising.utils.helper_functions import return_rx

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
    hyperparameters["resistance"] = float(args.resistance)
    hyperparameters["nb_flipping"] = int(args.nb_flipping)
    hyperparameters["cluster_threshold"] = float(args.cluster_threshold)
    hyperparameters["init_cluster_size"] = float(args.init_cluster_size)
    hyperparameters["end_cluster_size"] = float(args.end_cluster_size)

    # BRIM parameters
    dtBRIM = float(args.dtBRIM)
    hyperparameters["dtBRIM"] = dtBRIM
    hyperparameters["capacitance"] = float(args.capacitance)
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

def go_over_benchmark(which_benchmark: pathlib.Path, percentage:float=1.0, part:int=0) -> np.ndarray:
    """Go over all the benchmarks in the given directory.

    Args:
        which_benchmark (pathlib.Path): the path to the benchmark directory.

    Returns:
        np.ndarray: a list of all the benchmarks.
    """
    optimal_energies = which_benchmark / "optimal_energy.txt"
    benchmarks = np.loadtxt(optimal_energies, dtype=str)[:, 0]
    percentage = int(len(benchmarks) * percentage)
    if (part+1)*percentage == 1.0:
        return benchmarks[part*percentage:]
    else:
        return benchmarks[part*percentage:(part+1)*percentage]

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
            [
                "dtBRIM",
                "capacitance",
                "stop_criterion",
                "initial_temp_cont",
                "end_temp_cont",
                "seed",
                "coupling_annealing",
            ],
        ),
        "Multiplicative": (
            Multiplicative().solve,
            [
                "dtMult",
                "initial_temp_cont",
                "end_temp_cont",
                "seed",
                "coupling_annealing",
                "capacitance",
                "resistance",
                "flipping",
                "flipping_freq",
                "flipping_prob",
                "mu_param",
            ],
        ),
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
