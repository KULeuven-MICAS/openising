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

def make_directory(path:pathlib.Path) -> None:
    """Makes the given directory if it does not exist.

    Args:
        path (pathlib.Path): the directory to create
    """
    path.mkdir(parents=True, exist_ok=True)

def parse_hyperparameters(args:dict, num_iter:int) -> dict[str:]:
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

    # BRIM parameters
    dtBRIM = float(args.dtBRIM)
    hyperparameters["dtBRIM"] = dtBRIM
    hyperparameters["C"] = float(args.C)
    hyperparameters["flip"] = bool(args.flip)
    hyperparameters["stop_criterion"] = float(args.stop_criterion)

    # SA parameters
    hyperparameters["T"] = float(args.T)
    Tfin = float(args.T_final)
    hyperparameters["r_T"] = return_rx(num_iter, hyperparameters["T"], Tfin)

    # SCA parameters
    hyperparameters["q"] = float(args.q)

    # SB parameters
    hyperparameters["dtSB"] = float(args.dtSB)
    hyperparameters["a0"] = float(args.a0)
    hyperparameters["c0"] = float(args.c0)
    hyperparameters["seed"] = int(args.seed)

    return hyperparameters

def get_best_found_gurobi(gurobi_files:list[pathlib.Path]) -> list[float]:
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
    clock_freq: float=1e6,
    clock_op: int=1000,
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
        tuple[np.ndarray, float]: optimal state and energy of the specified solver.
    """
    optim_state = np.zeros((model.num_variables,))
    optim_energy = None
    if solver == "BRIM":
        v = 0.1 * s_init
        optim_state, optim_energy = BRIM().solve(
            model=model,
            v=v,
            num_iterations=num_iter,
            dt=hyperparameters["dtBRIM"],
            C=hyperparameters["C"],
            file=logfile,
            random_flip=hyperparameters["flip"],
            stop_criterion=hyperparameters["stop_criterion"],
            seed=hyperparameters["seed"],
            Temp=hyperparameters["T"],
            r_T=hyperparameters["r_T"],
        )
    elif solver == "Multiplicative":
        v = 0.5*s_init
        optim_state, optim_energy = Multiplicative().solve(
            model=model,
            v=v,
            num_iterations=num_iter,
            dt=hyperparameters["dtMult"],
            logfile=logfile
        )
    elif solver == "SA":
        optim_state, optim_energy = SASolver().solve(
            model=model,
            initial_state=s_init,
            num_iterations=num_iter,
            initial_temp=hyperparameters["T"],
            cooling_rate=hyperparameters["r_T"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver == "DSA":
        optim_state, optim_energy = DSASolver().solve(
            model=model,
            initial_state=s_init,
            num_iterations=num_iter,
            initial_temp=hyperparameters["T"],
            cooling_rate=hyperparameters["r_T"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver == "SCA":
        optim_state, optim_energy = SCA().solve(
            model=model,
            sample=s_init,
            num_iterations=num_iter,
            T=hyperparameters["T"],
            r_t=hyperparameters["r_T"],
            q=hyperparameters["q"],
            r_q=hyperparameters["r_q"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver[1:] == "SB":
        x = s_init*np.arange(0.01/model.num_variables, 0.01+0.01/model.num_variables, 0.01/model.num_variables)
        y = np.zeros((model.num_variables,))
        dt = hyperparameters["dtSB"]
        c0 = hyperparameters["c0"]
        a0 = hyperparameters["a0"]
        if solver[0] == "b":
            optim_state, optim_energy = ballisticSB().solve(
                model=model,
                x=x,
                y=y,
                num_iterations=num_iter,
                c0=c0,
                dt=dt,
                a0=a0,
                file=logfile,
                clock_freq=clock_freq, clock_op=clock_op,
            )
        elif solver[0] == "d":
            optim_state, optim_energy = discreteSB().solve(
                model=model,
                x=x,
                y=y,
                num_iterations=num_iter,
                c0=c0,
                dt=dt,
                a0=a0,
                file=logfile,
                clock_freq=clock_freq, clock_op=clock_op,
            )
    return optim_state, optim_energy


def return_rx(num_iter: int, r_init:float, r_final:float) -> float:
    """Returns the change rate of SA/SCA hyperparameters

    Args:
        num_iter (int): amount of iterations.
        r_init (float): the initial value of the hyperparameter.
        r_final (float): the end value of the hyperparameter.

    Returns:
        float: the change rate of the hyperarameter.
    """
    return (r_final/r_init)**(1/(num_iter + 1))

def return_c0(model: IsingModel)->float:
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

def return_G(problem: IsingModel)->float:
    """Returns the optimal latch resistant value for the given problem.

    Args:
        problem (IsingModel): the problem that will be solved with BRIM.

    Returns:
        float: the latch resistance.
    """
    sumJ = np.sum(np.abs(triu_to_symm(problem.J)), axis=0)
    return np.average(sumJ)*2

def return_q(problem: IsingModel)->float:
    """Returns the optimal value for the penalty parameter q for the SCA solver.

    Args:
        problem (IsingModel): the problem that will be solved with SCA.

    Returns:
        float: the penalty parameter q.
    """
    eig = np.abs(spalg.eigs(triu_to_symm(-problem.J), 1)[0][0])
    return eig / 2
