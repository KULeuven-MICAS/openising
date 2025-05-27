import argparse
import time
import numpy as np

from ising.flow import LOGGER, TOP
from ising.generators.Knapsack import random_Knapsack
from ising.utils.flow import parse_hyperparameters
from ising.utils.threading import make_solvers_thread, make_Gurobi_thread
from ising.utils.helper_functions import make_directory, return_c0, return_q, return_rx


def run_Knapsack_dummy(
    N_list: list[int], dens: int, num_iter: int, solvers: list[str], args: argparse.Namespace
) -> None:
    """Generates ands runs dummy Knapsack problems with the specified density.

    Args:
        N_list (list[int]): the sizes of the problems.
        dens (int): The density of the profit matrix.
        num_iter (int): the amount of iterations.
        solvers (list[str]): the list of solvers to run.
        args (argparse.Namespace): the parsed arguments from the command line.
    """
    nb_runs = int(args.nb_runs)

    seed = int(args.seed)
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    use_gurobi = bool(args.use_gurobi)

    num_iter = int(args.num_iter)
    hyperparameters = parse_hyperparameters(args, num_iter)

    # SCA parameters
    if hyperparameters["q"] == 0.0:
        change_q = True
        hyperparameters["r_q"] = 1.0
    else:
        hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
        change_q = False

    # SB parameters
    if hyperparameters["c0"] == 0.0:
        change_c = True
    else:
        change_c = False

    logtop = TOP / "ising/flow/MaxCut/logs"
    LOGGER.debug("Logpath: " + str(logtop))
    make_directory(logtop)
    figtop = TOP / "ising/flow/MaxCut/plots" / str(args.fig_folder)
    make_directory(figtop)

    problems = {}
    for N in N_list:
        problem = random_Knapsack(N, dens, penalty_value=float(args.penalty_val), bit_width=int(args.bit_width))
        problems[N] = problem

    if use_gurobi:
        logfiles = {}
        for N in N_list:
            logfile = logtop / f"Gurobi_Knapsack_dens{dens}_N{N}.log"
            logfiles[N] = logfile
        make_Gurobi_thread(models=problems, logfiles=logfiles)

    for N in N_list:
        LOGGER.info(f"Solving with {N} variables")
        problem = problems[N]
        if change_c:
            hyperparameters["c0"] = return_c0(model=problem)
        if change_q:
            hyperparameters["q"] = return_q(problem)

        logfiles = {
            solver: [logtop / f"{solver}_Knapsack_N{N}_dens{dens}_run{run}.log" for run in range(nb_runs)]
            for solver in solvers
        }
        make_solvers_thread(
            solvers=solvers,
            num_iter=num_iter,
            model=problem,
            nb_runs=nb_runs,
            logfiles=logfiles,
            **hyperparameters,
        )
