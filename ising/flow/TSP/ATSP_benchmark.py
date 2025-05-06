import argparse
import numpy as np

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.ATSP import ATSP_parser
from ising.generators.TSP import TSP
from ising.flow.TSP.Calculate_TSP_energy import calculate_TSP_energy
from ising.utils.flow import run_solver, make_directory, parse_hyperparameters, return_q, return_c0
# from ising.utils.threading import make_solvers_thread
from ising.solvers.Gurobi import Gurobi
from ising.postprocessing.TSP_plot import plot_graph_solution


def run_ATSP_benchmark(benchmark: str, iter_list: list[int], solvers: list[str], args: argparse.Namespace):
    """Runs the given TSP benchmark on different solvers.
    Each solver is run nb_run times with the specified number of iterations.

    Args:
        benchmark (str): the TSP benchmark to run
        iter_list (tuple[int]): A list containing the iteration lengths.
        solvers (list[str]): a list with all the solvers to run.
        args (_type_): the arguments parsed with ising/flow/Problem_parser.py
    """
    LOGGER.info("Generating benchmark: " + str(benchmark))
    graph_orig, best_found = ATSP_parser(benchmark=TOP / f"ising/benchmarks/ATSP/{benchmark}.atsp")
    A = float(args.weight_constant)
    model = TSP(graph=graph_orig, weight_constant=A)
    if best_found is not None:
        LOGGER.info(f"Best found: {best_found}")
    LOGGER.info("Generated benchmark")

    nb_runs = int(args.nb_runs)
    logpath = TOP / "ising/flow/TSP/logs_ATSP"
    figtop = TOP / "ising/flow/TSP/plots_ATSP" / args.fig_folder
    LOGGER.debug(f"Logpath: {logpath}")
    LOGGER.debug(f"Figtop: {figtop}")
    make_directory(logpath)
    make_directory(figtop)

    if bool(args.use_gurobi):
        gurobi_log = logpath / f"Gurobi_{benchmark}.log"
        Gurobi().solve(model=model, file=gurobi_log)
        calculate_TSP_energy([gurobi_log], graph_orig, gurobi=True)
        plot_graph_solution(
            fileName=gurobi_log, G_orig=graph_orig, save_folder=figtop, fig_name=f"Gurobi_{benchmark}_graph.png"
        )

    for num_iter in iter_list:
        LOGGER.info(f"Running for {num_iter} iterations")
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

        for run in range(nb_runs):
            init_state = np.random.uniform(-1, 1, (model.num_variables,))
            for solver in solvers:
                logfile = logfiles[solver][run]
                run_solver(solver, num_iter, init_state, model, logfile, **hyperparameters)

        # make_solvers_thread(
        #     solvers, model=model, num_iter=num_iter, nb_runs=nb_runs, logfiles=logfiles, **hyperparameters
        # )

        calculate_TSP_energy(np.array([logfile for (solver, logfile) in logfiles.items()]).flatten(), graph_orig)
        # for solver in solvers:
        #     plot_graph_solution(
        #         fileName=logfiles[solver][-1],
        #         G_orig=graph_orig,
        #         save_folder=figtop,
        #         fig_name=f"{solver}_{benchmark}_graph.png",
        #     )
