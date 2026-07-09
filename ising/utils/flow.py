import pathlib
import numpy as np
from argparse import Namespace
import scipy.sparse.linalg as spalg

from ising.utils.HDF5Logger import return_metadata

from ising.stages.model.ising import IsingModel
from ising.utils.helper_functions import return_rx
from ising.utils.numpy import triu_to_symm


def parse_hyperparameters(args: Namespace) -> dict[str:]:
    """Parses the arguments needed for the solvers.

    Args:
        args (dict): the command line arguments.
        num_iter (int): amount of iterations

    Returns:
        dict[str:Any]: the hyperparameters for the solvers.
    """
    hyperparameters = dict()

    # seed hyperparameter
    hyperparameters["seed"] = int(args.seed)

    if "Multiplicative" in args.solvers or "BRIM" in args.solvers:
        hyperparameters["capacitance"] = float(args.capacitance)
        hyperparameters["stop_criterion"] = float(args.stop_criterion)

    # Multiplicative parameters
    if "Multiplicative" in args.solvers:
        hyperparameters["num_iterations_Multiplicative"] = int(args.num_iterations_Multiplicative)
        hyperparameters["nb_flipping"] = int(args.nb_flipping)
        hyperparameters["cluster_threshold"] = float(args.cluster_threshold)
        hyperparameters["init_cluster_size"] = float(args.init_cluster_size)
        hyperparameters["end_cluster_size"] = float(args.end_cluster_size)
        hyperparameters["cluster_choice"] = args.cluster_choice
        hyperparameters["exponent"] = float(args.exponent)
        hyperparameters["accumulation_delay"] = float(args.accumulation_delay)
        hyperparameters["broadcast_delay"] = float(args.broadcast_delay)
        hyperparameters["delay_offset"] = float(args.delay_offset)
        hyperparameters["current"] = float(args.current)
        # hyperparameters["sigma_J"] = float(args.sigma_J)
        hyperparameters["combine_nodes"] = bool(args.combine_nodes)
        hyperparameters["nb_splits"] = int(args.nodes_scaling) if hasattr(args, "nodes_scaling") else 1

    # BRIM parameters
    if "BRIM" in args.solvers:
        hyperparameters["resistance"] = float(args.resistance)
        hyperparameters["num_iterations_BRIM"] = int(args.num_iterations_BRIM)
        hyperparameters["dtBRIM"] = float(args.dtBRIM)
        hyperparameters["do_flipping"] = bool(args.do_flipping)
        hyperparameters["coupling_annealing"] = bool(args.coupling_annealing)
        hyperparameters["probability_start"] = float(args.probability_start)

    # SA-like parameters
    if "SA" in args.solvers or "DSA" in args.solvers:
        hyperparameters["initial_temp"] = float(args.T)

    if "SA" in args.solvers:
        hyperparameters["num_iterations_SA"] = int(args.num_iterations_SA)
        Tfin = float(args.T_final)
        hyperparameters["cooling_rate_SA"] = (
            return_rx(hyperparameters["num_iterations_SA"], hyperparameters["initial_temp"], Tfin)
            if hyperparameters["initial_temp"] != 0
            else 1.0
        )
    if "DSA" in args.solvers:
        hyperparameters["num_iterations_DSA"] = int(args.num_iterations_DSA)
        Tfin = float(args.T_final)
        hyperparameters["cooling_rate_DSA"] = (
            return_rx(hyperparameters["num_iterations_DSA"], hyperparameters["initial_temp"], Tfin)
            if hyperparameters["initial_temp"] != 0
            else 1.0
        )


    # in-Situ SA parameters
    if "inSituSA" in args.solvers:
        hyperparameters["initial_temp_inSituSA"] = float(args.initial_temp_inSituSA)
        hyperparameters["num_iterations_inSituSA"] = int(args.num_iterations_inSituSA)
        hyperparameters["nb_flips"] = int(args.nb_flips)
        Tfin = float(args.T_final_inSituSA)
        hyperparameters["cooling_rate_inSituSA"] = (
            return_rx(hyperparameters["num_iterations_inSituSA"], hyperparameters["initial_temp_inSituSA"], Tfin)
            if hyperparameters["initial_temp_inSituSA"] != 0
            else 1.0
        )

    # SCA parameters
    if "SCA" in args.solvers:
        hyperparameters["num_iterations_SCA"] = int(args.num_iterations_SCA)
        hyperparameters["q"] = float(args.q)
        if float(args.q_final) != -1:
            hyperparameters["r_q"] = (
                return_rx(hyperparameters["num_iterations_SCA"], hyperparameters["q"], float(args.q_final))
                if hyperparameters["q"] != 0
                else 1.0
            )
        else:
            hyperparameters["r_q"] = 1.0
        hyperparameters["initial_temp_SCA"] = float(args.T_init_SCA)
        Tfin = float(args.T_final_SCA)
        hyperparameters["cooling_rate_SCA"] = (
            return_rx(hyperparameters["num_iterations_SCA"], hyperparameters["initial_temp_SCA"], Tfin)
            if hyperparameters["initial_temp_SCA"] != 0
            else 1.0
        )

    # SB parameters
    if "dSB" in args.solvers or "bSB" in args.solvers:
        hyperparameters["num_iterations_SB"] = int(args.num_iterations_SB)
        hyperparameters["a0"] = float(args.a0)
        hyperparameters["c0"] = float(args.c0)
    if "dSB" in args.solvers:
        hyperparameters["dtdSB"] = float(args.dtdSB)
    if"bSB" in args.solvers:
        hyperparameters["dtbSB"] = float(args.dtbSB)

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


def go_over_benchmark(which_benchmark: pathlib.Path, percentage: float = 1.0, part: int = 0) -> np.ndarray:
    """Go over all the benchmarks in the given directory.

    Args:
        which_benchmark (pathlib.Path): the path to the benchmark directory.

    Returns:
        np.ndarray: a list of all the benchmarks.
    """
    optimal_energies = which_benchmark / "optimal_energy.txt"
    benchmarks = np.loadtxt(optimal_energies, dtype=str)[:, 0]
    percentage = int(len(benchmarks) * percentage)
    if (part + 1) * percentage == 1.0:
        return benchmarks[part * percentage :]
    else:
        return benchmarks[part * percentage : (part + 1) * percentage]


def return_c0(model: IsingModel) -> float:
    """Returns the optimal c0 value for simulated bifurcation.

    Args:
        model (IsingModel): the Ising model that will be solved with simulated Bifurcationl.

    Returns:
        float: the c0 hyperaparameter.
    """
    denominator = np.sqrt(model.num_variables) * np.sqrt(
        np.sum(np.power(model.J, 2)) / (model.num_variables * (model.num_variables - 1))
    )
    if np.isclose(denominator, 0.0, atol=1e-12):
        return 0.5
    else:
        return 0.5 / denominator


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


def relative_to_best_found(energy: np.ndarray[float], best_found: float) -> np.ndarray[float]:
    """Computes the relative distance from the best found energy. The return value will always be between 0 and 1,
    no matter if the best found is positive or negative

    @param energy (float): the energy to compute the relative distance for.
    @param best_found (float): the baseline energy.

    @return np.ndarray[float]: the relative distance.
    """
    if best_found != 0:
        return np.abs(energy-best_found) / np.abs(best_found)
    else:
        return energy

def approximation_to_best_found(energy: np.ndarray[float], best_found:float) -> np.ndarray[float]:
    """Computes the approximation to the best found energy in percentage.

    @param energy (float): the energy to compute the approximation for.
    @param best_found (float): the baseline energy.

    @return np.ndarray[float]: the approximation in percentage.
    """
    if best_found != 0.0:
        return 100*(1 - relative_to_best_found(energy, best_found))
    else:
        return 1/np.array([en if en != 0 else 1 for en in energy]) * 100

def compute_ttt(energies:np.ndarray, computation_time:float, best_found:float, nb_runs:int, target:float=0.9) -> float:
    """Computes the time to target of a set of energies and the average computation time.

    @param energies (np.ndarray): set of solution energies
    @param computation_time (float): average computation time over all runs.
    @param best_found (float): best reported solution of problem
    @param nb_runs (int): amount of runs the solver went through.
    @param target (float): target value from best found. Default set to 0.9.

    @returns ttt (float): time to target.
    """
    result = computation_time * np.log(1 - 0.99)
    rel_error = relative_to_best_found(np.array(energies), best_found)
    P_s = np.sum(rel_error <= (1-target)) / nb_runs
    if P_s >= 0.99:
        denom = np.log(1 - 0.99)
    else:
        denom = np.log(1-P_s)
    return result / denom
