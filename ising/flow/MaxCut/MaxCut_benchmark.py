import argparse
import numpy as np

from ising.flow import LOGGER, TOP
from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.solvers.Gurobi import Gurobi
from ising.utils.flow import make_directory, parse_hyperparameters, return_c0, return_rx, return_q, run_solver


def run_benchmark(benchmark:str, iter_list:list[int], solvers:list[str], args:argparse.Namespace) -> None:
    """Runs a given benchmark with the specified list of iteration lengths.
    It is important the arguments are parsed using ising/flow/Problem_parser.py

    Args:
        benchmark (str): the benchmark to run
        iter_list (tuple[int]): the list of iterations lengths.
        solvers (list[str]): the list of solvers to run.
    """
    LOGGER.info("Generating benchmark: " + benchmark)
    graph, best_found = G_parser(benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt")
    model = MaxCut(graph=graph)
    if best_found is not None:
        LOGGER.info("Best found energy: " +  str(-best_found))
    LOGGER.info("Generated benchmark")

    nb_runs = int(args.nb_runs)

    LOGGER.info("Setting up solvers")
    logpath = TOP / "ising/flow/MaxCut/logs"
    LOGGER.debug("Logpath: "+ str(logpath))
    make_directory(logpath)

    if bool(int(args.use_gurobi)):
        gurobi_log = logpath / f"Gurobi_{benchmark}.log"
        Gurobi().solve(model=model, file=gurobi_log)

    for num_iter in iter_list:
        LOGGER.info(f"Running for {num_iter} iterations")
        hyperparameters = parse_hyperparameters(args, num_iter)

        if hyperparameters["c0"] == 0.0:
            hyperparameters["c0"] = return_c0(model=model)
        if hyperparameters["q"] == 0.0:
            hyperparameters["q"] = return_q(model)
            hyperparameters["r_q"] = 1.0
        else:
            hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
        for run in range(nb_runs):
            hyperparameters["seed"] = run + 1 + int(args.seed)
            initial_state = np.random.uniform(-1, 1, (model.num_variables,))
            for solver in solvers:
                logfile = logpath / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
                run_solver(solver, num_iter, initial_state, model, logfile, **hyperparameters)
