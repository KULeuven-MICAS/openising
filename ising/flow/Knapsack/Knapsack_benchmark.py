import argparse
import numpy as np

from ising.flow import LOGGER, TOP
from ising.benchmarks.parsers.Knapsack import QKP_parser
from ising.generators.Knapsack import knapsack
from ising.solvers.Gurobi import Gurobi
from ising.utils.flow import parse_hyperparameters, run_solver
from ising.utils.helper_functions import make_directory, return_c0, return_q, return_rx


def run_Knapsack_benchmark(size: int, dens: int, num_iter: int, solvers: list[str], args: argparse.Namespace) -> None:
    """Runs a given benchmark with the specified list of iteration lengths.
    It is important the arguments are parsed using ising/flow/Problem_parser.py

    Args:
        size (int): the size of the benchmark.
        dens (int): the density of the benchmark.
        num_iter (int): The amount of iterations.
        solvers (list[str]): the list of solvers to run.
    """
    LOGGER.info(f"Generating benchmark: jeu_{size}_{dens}_1-5")
    models = dict()
    bench_top = TOP / "ising/benchmarks/Knapsack"
    for i in range(5):
        benchmark = bench_top / f"jeu_{size}_{dens}_{i + 1}"
        profit, weight, capacity, best_found = QKP_parser(benchmark=benchmark)
        models[f"jeu_{size}_{dens}_{i + 1}"] = (knapsack(profit, capacity, weight, float(args.penalty_val)), best_found)
        if best_found is not None:
            LOGGER.info("Best found energy: " + str(-best_found))
    LOGGER.info("Generated benchmark")

    nb_runs = int(args.nb_runs)

    LOGGER.info("Setting up solvers")
    logpath = TOP / "ising/flow/MaxCut/logs"
    LOGGER.debug("Logpath: " + str(logpath))
    make_directory(logpath)

    hyperparameters = parse_hyperparameters(args, num_iter)

    for benchmark, mod in models.items():
        model, best_found = mod
        if bool(int(args.use_gurobi)):
            gurobi_log = logpath / f"Gurobi_{benchmark}.log"
            Gurobi().solve(model=model, file=gurobi_log)

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
