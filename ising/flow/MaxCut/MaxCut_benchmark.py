import os
import pathlib
import argparse
# import numpy as np

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.solvers.Gurobi import Gurobi
from ising.utils.flow import make_directory, parse_hyperparameters, return_c0, return_rx, return_q#, run_solver
from ising.utils.threading import make_solvers_thread

TOP = pathlib.Path(os.getenv("TOP"))

def run_benchmark(benchmark:str, iter_list:list[int], solvers:list[str], args:argparse.Namespace) -> None:
    """Runs a given benchmark with the specified list of iteration lengths.
    It is important the arguments are parsed using ising/flow/Problem_parser.py

    Args:
        benchmark (str): the benchmark to run
        iter_list (tuple[int]): the list of iterations lengths.
        solvers (list[str]): the list of solvers to run.
    """
    print("Generating benchmark: ", benchmark)
    graph, best_found = G_parser(benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt")
    model = MaxCut(graph=graph)
    if best_found is not None:
        print("Best found energy: ", -best_found)
    print("Generated benchmark")

    # iter_list = np.array(range(iter_list[0], iter_list[1], 100))
    nb_runs = int(args.nb_runs)

    print("Setting up solvers")
    logpath = TOP / "ising/flow/MaxCut/logs"
    make_directory(logpath)

    if bool(int(args.use_gurobi)):
        gurobi_log = logpath / f"Gurobi_{benchmark}.log"
        Gurobi().solve(model=model, file=gurobi_log)

    for num_iter in iter_list:
        print(f"Running for {num_iter} iterations")
        hyperparameters = parse_hyperparameters(args, num_iter)

        if hyperparameters["c0"] == 0.0:
            hyperparameters["c0"] = return_c0(model=model)
        if hyperparameters["q"] == 0.0:
            hyperparameters["q"] = return_q(model)
            hyperparameters["r_q"] = 1.0
        else:
            hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
        logfiles = {}
        for solver in solvers:
            logfiles[solver] = []
            for run in range(nb_runs):
                logfile = logpath / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
                logfiles[solver].append(logfile)
                # s_init = np.random.choice([-1,1], (model.num_variables,))
                # run_solver(solver, num_iter, s_init, model, logfile, **hyperparameters)

        make_solvers_thread(
            solvers, model=model, num_iter=num_iter, nb_runs=nb_runs, logfiles=logfiles, **hyperparameters
        )
