import pathlib
import os
import argparse

from ising.benchmarks.parsers.ATSP import ATSP_parser
from ising.generators.TSP import TSP
from ising.flow.TSP.Calculate_TSP_energy import calculate_TSP_energy
from ising.utils.flow import make_directory, parse_hyperparameters, return_q, return_c0
from ising.utils.threading import make_solvers_thread

TOP = pathlib.Path(os.getenv("TOP"))

def run_TSP_benchmark(benchmark:str, iter_list:list[int], solvers:list[str], args:argparse.Namespace):
    """Runs the given TSP benchmark on different solvers.
    Each solver is run nb_run times with the specified number of iterations.

    Args:
        benchmark (str): the TSP benchmark to run
        iter_list (tuple[int]): A list containing the iteration lengths.
        solvers (list[str]): a list with all the solvers to run.
        args (_type_): the arguments parsed with ising/flow/Problem_parser.py
    """
    print("Generating benchmark: ", benchmark)
    graph_orig, best_found = ATSP_parser(benchmark=TOP / f"ising/benchmarks/ATSP/{benchmark}.txt")
    A = float(args.weight_constant)
    B = float(args.place_constraint)
    C = float(args.time_constraint)
    model = TSP(graph=graph_orig, A=A, B=B, C=C)
    if best_found is not None:
        print(f"Best found: {best_found}")
    print("Generated benchmark")

    nb_runs = int(args.nb_runs)
    logpath = TOP / "ising/flow/TSP/logs"
    make_directory(logpath)

    for num_iter in iter_list:
        print(f"Running for {num_iter} iterations")
        hyperparameters = parse_hyperparameters(args, num_iter)
        if hyperparameters["q"] == 0.0:
            hyperparameters["q"] = return_q(model)
            hyperparameters["r_q"] = 1.0
        if hyperparameters["c0"] == 0.0:
            hyperparameters["c0"] = return_c0(model=model)

        logfiles = {}
        for solver in solvers:
            logfiles[solver] = []
            for run in range(nb_runs):
                logfile = logpath / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
                logfiles[solver].append(logfile)

        make_solvers_thread(
            solvers, model=model, num_iter=num_iter, nb_runs=nb_runs, logfiles=logfiles, **hyperparameters
        )

        calculate_TSP_energy([logfile for (solver, logfile) in logfiles.items()].flatten(), graph_orig)
