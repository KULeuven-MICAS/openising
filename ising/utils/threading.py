import numpy as np  # noqa: A005
import pathlib
import threading


from ising.model.ising import IsingModel
from ising.solvers.Gurobi import Gurobi
from ising.utils.helper_solvers import run_solver


def solver_thread(
    solver: str,
    sample: np.ndarray,
    num_iter: int,
    model: IsingModel,
    nb_runs,
    logfiles: list[pathlib.Path],
    **hyperparameters,
) -> None:
    """Runs the solver for the given amount of runs. This function can be used for threading.

    Args:
        solver (str): the solver to use.
        sample (np.ndarray): the sample to start the solver with.
        num_iter (int): the amount of iterations.
        model (IsingModel): the model to solve.
        nb_runs (_type_): the amount of runs.
        logfiles (list[pathlib.Path]): the logfiles to store the data.
    """
    for run in nb_runs:
        logfile = logfiles[run]
        run_solver(solver=solver, num_iter=num_iter, s_init=sample, model=model, logfile=logfile, **hyperparameters)

def make_solvers_thread(
    solvers: list[str],
    sample: np.ndarray,
    num_iter: int,
    model: IsingModel,
    nb_runs: int,
    logfiles: dict[str : list[pathlib.Path]],
    **hyperparameters,
) -> list[threading.Thread]:
    """Makes a thread for each solver and starts them.

    Args:
        solvers (list[str]): the list of solvers.
        sample (np.ndarray): sample at which the solver should start.
        num_iter (int): amount of iterations for the solvers.
        model (IsingModel): the model that needs to be solved.
        nb_runs (int): the amount of runs for each solver.
        logfiles (dict[str:list[pathlib.Path]]): a dictionary of all the possible logfiles.

    Returns:
       list[threading.Thread]: the list of the threads on which the main thread should wait.
    """
    threads = []
    for solver in solvers:
        thread = threading.Thread(
            target=solver_thread,
            args=(solver, sample, num_iter, model, nb_runs, logfiles[solver]),
            kwargs=hyperparameters,
        )
        thread.start()
        threads.append(thread)
    return threads

def solve_Gurobi(model:IsingModel, result: tuple):
    """Solves the given model with Gurobi and stores the results in the given tuple.
    This function can be used for threading.

    Args:
        model (IsingModel): the model to solve with Gurobi.
        result (tuple): the tuple in which the results are stored.
    """
    sample, energy = Gurobi().solve(model=model)
    result[0] = sample
    result[1] = energy

def make_Gurobi_thread(models:dict[int:IsingModel], results:dict[int:tuple]) -> list[threading.Thread]:
    """Generates multiple a thread for each model that is given which will solve the problem with Gurobi.

    Args:
        models (dict[int:IsingModel]): dictionary of all the models with the size as its key.
        results (dict[int:tuple]): dictionary where the results of the optimisation will be stored.

    Returns:
        list[threading.Thread]: the resulting threads.
    """
    threads = []
    for N, model in models.items():
        result = results[N]
        thread = threading.Thread(target=solve_Gurobi, args=(model, result))
        thread.start()
        threads.append(thread)
    return threads
