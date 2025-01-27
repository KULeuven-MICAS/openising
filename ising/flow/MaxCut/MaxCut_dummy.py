import os
import pathlib
import numpy as np
import networkx as nx
import time
import argparse

from ising.generators.MaxCut import random_MaxCut

from ising.utils.flow import return_c0, return_rx, return_q, make_directory, parse_hyperparameters
from ising.postprocessing.MC_plot import plot_MC_solution
from ising.utils.threading import make_solvers_thread, make_Gurobi_thread

TOP = pathlib.Path(os.getenv("TOP"))

def run_dummy(N_list:list[int], solvers:list[str], args:argparse.Namespace) -> None:
    """Runs some dummy Max-Cut problems with the specified size of the problems.

    Args:
        N_list (list[int]): list containing the problem sizes
        solvers (list[str]): list of solvers to run
    """

    nb_runs = int(args.nb_runs)

    seed = int(args.seed)
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    fig_name = str(args.fig_name)
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
    make_directory(logtop)
    figtop = TOP / "ising/flow/MaxCut/plots" / str(args.fig_folder)
    make_directory(figtop)


    problems = {}
    for N in N_list:
        problem = random_MaxCut(N)
        problems[N] = problem

    if use_gurobi:
        logfiles = {}
        for N in N_list:
            logfile = logtop / f"Gurobi_N{N}.log"
            logfiles[N] = logfile
        make_Gurobi_thread(nb_cores=3, models=problems, logfiles=logfiles)


    for N in N_list:
        print(f"Solving with {N} variables")
        problem = problems[N]
        if change_c:
            hyperparameters["c0"] = return_c0(model=problem)
        if change_q:
            hyperparameters["q"] = return_q(problem)
        logfiles = {}
        # print(np.sum(problem.J, axis=1))
        for solver in solvers:
            logfiles[solver] = []
            for nb_run in range(nb_runs):
                logfiles[solver].append(logtop / f"{solver}_N{N}_run{nb_run}.log")
        make_solvers_thread(
            nb_cores=len(solvers),
            solvers=solvers,
            num_iter=num_iter,
            model=problem,
            nb_runs=nb_runs,
            logfiles=logfiles,
            **hyperparameters,
        )
        if N <= 20:
            G_orig = nx.Graph()
            G_orig.add_nodes_from(list(range(N)))
            for i in range(N):
                for j in range(i+1, N):
                    if problem.J[i, j] != 0:
                        G_orig.add_edge(i, j)

            for solver in solvers:
                plot_MC_solution(fileName=logfiles[solver][-1], G_orig=G_orig, save_folder=figtop,
                                fig_name=f"{solver}_N{N}_graph_{fig_name}")
    print("Done")
