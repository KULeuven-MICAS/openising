import numpy as np  # noqa: A005
import pathlib
from multiprocessing import Pool
from functools import partial

from ising.flow import LOGGER
from ising.model.ising import IsingModel
from ising.solvers.Gurobi import Gurobi
from ising.utils.flow import run_solver


def solver_thread(
    solver: str,
    num_iter: int,
    model: IsingModel,
    nb_runs:int,
    logfiles: list[pathlib.Path],
    **hyperparameters,
) -> None:
    """Runs the solver for the given amount of runs. This function can be used for threading.

    Args:
        solver (str): the solver to use.
        num_iter (int): the amount of iterations.
        model (IsingModel): the model to solve.
        nb_runs (_type_): the amount of runs.
        logfiles (list[pathlib.Path]): the logfiles to store the data.
    """
    for run in range(nb_runs):
        s_init = np.random.uniform(-1, 1, (model.num_variables,))
        logfile = logfiles[run]
        run_solver(solver=solver, num_iter=num_iter, s_init=s_init, model=model,
                                               logfile=logfile, **hyperparameters)
    LOGGER.info(f"{solver} done")

def make_solvers_thread(
    solvers: list[str],
    num_iter: int,
    model: IsingModel,
    nb_runs: int,
    logfiles: dict[str : list[pathlib.Path]],
    **hyperparameters,
) -> None:
    """Makes a thread for each solver and starts them.

    Args:
        solvers (list[str]): the list of solvers.
        num_iter (int): amount of iterations for the solvers.
        model (IsingModel): the model that needs to be solved.
        nb_runs (int): the amount of runs for each solver.
        logfiles (dict[str:list[pathlib.Path]]): a dictionary of all the possible logfiles.
    """
    pool = Pool(processes=len(solvers))
    pool.starmap(partial(solver_thread, **hyperparameters),
        iterable=[(solver, num_iter, model, nb_runs, logfiles[solver]) for solver in solvers],
    )


def solve_Gurobi(model: IsingModel, file: pathlib.Path) -> None:
    """Solves the given model with Gurobi and stores the results in the given tuple.
    This function can be used for threading.

    Args:
        model (IsingModel): the model to solve with Gurobi.
        file (pathlib.Path): The logfile in which the results are stored.
    """
    LOGGER.info(f"Solving with Gurobi with {model.num_variables} variables")
    state, energy = Gurobi().solve(model=model, file=file)
    LOGGER.info(f"Optimal {state=}, with {energy=}")


def make_Gurobi_thread(models: dict[int:IsingModel], logfiles: dict[int : pathlib.Path],) -> None:
    """Generates multiple a thread for each model that is given which will solve the problem with Gurobi.

    Args:
        models (dict[int:IsingModel]): dictionary of all the models with the size as its key.
        logfiles (dict[int:pathlib.Path]): dictionary where the logfiles are stored.

    """
    pool = Pool(processes=len(models.keys()))
    pool.starmap(func=solve_Gurobi, iterable=[(models[N], logfiles[N]) for N in models.keys()])
