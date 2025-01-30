import argparse
import os
import pathlib
import time
import numpy as np

from ising.generators.TSP import generate_random_TSP
from ising.flow.TSP.Calculate_TSP_energy import calculate_TSP_energy
from ising.utils.threading import make_solvers_thread#, make_Gurobi_thread
from ising.utils.flow import make_directory, parse_hyperparameters, return_q, return_c0, return_rx
from ising.postprocessing.TSP_plot import plot_graph_solution
from ising.utils.HDF5Logger import return_metadata
from ising.solvers.Gurobi import Gurobi

TOP = pathlib.Path(os.getenv("TOP"))


def run_TSP_dummy(N_list: list[int], solvers: list[str], args: argparse.Namespace):
    """Generates random TSP problems of sizes in the N_list and runs the specified solvers on them.

    Args:
        N_list (list[int]): the sizes of the TSP problems to generate
        solvers (list[str]): the solvers to run
        args (argparse.Namespace): the arguments parsed with ising/flow/Problem_parser.py
    """
    logpath = TOP / "ising/flow/TSP/logs"
    figtop = TOP / "ising/flow/TSP/plots" / args.fig_folder
    make_directory(logpath)
    make_directory(figtop)

    nb_runs = int(args.nb_runs)
    seed = int(args.seed)
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    use_gurobi = bool(args.use_gurobi)
    num_iter = int(args.num_iter)
    hyperparameters = parse_hyperparameters(args, num_iter)

    if hyperparameters["q"] == 0.0:
        change_q = True
        hyperparameters["r_q"] = 1.0
    else:
        change_q = False
        hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))

    if hyperparameters["c0"] == 0.0:
        change_c = True
    else:
        change_c = False

    problems = {}
    graphs = {}
    weight_constant = float(args.weight_constant)
    for N in N_list:
        problems[N], graphs[N] = generate_random_TSP(N, seed, weight_constant)

    if use_gurobi:
        logfiles = {N: logpath / f"Gurobi_N{N}.log" for N in N_list}

        # make_Gurobi_thread(models=problems, logfiles=logfiles)

        for N in N_list:
            state, energy = Gurobi().solve(model=problems[N], file=logfiles[N])
            print(f"Optimal {state=} with {energy=}")
            calculate_TSP_energy([logfiles[N]], graphs[N], gurobi=True)
            plot_graph_solution(fileName=logfiles[N], G_orig=graphs[N], save_folder=figtop,
                                fig_name=f"Gurobi_N{N}_graph.png")


    for N in N_list:
        print(f"Running for {N} cities")
        # print(f"The problem that will be solved is: {problems[N]}")
        if change_q:
            hyperparameters["q"] = return_q(problems[N])
        if change_c:
            hyperparameters["c0"] = return_c0(problems[N])

        logfiles = {solver: [logpath / f"{solver}_N{N}_run{run}.log" for run in range(nb_runs)] for solver in solvers}
        make_solvers_thread(
            solvers,
            num_iter=num_iter,
            model=problems[N],
            nb_runs=nb_runs,
            logfiles=logfiles,
            **hyperparameters
        )
        calculate_TSP_energy(np.array([logfile for (_, logfile) in logfiles.items()]).flatten(), graphs[N])
        if N <= 20:
            for solver in solvers:
                plot_graph_solution(fileName=logfiles[solver][-1], G_orig=graphs[N], save_folder=figtop,
                                fig_name=f"{solver}_N{N}_graph.png")
                solution_state = return_metadata(fileName=logfiles[solver][-1], metadata="solution_state")
                print(f"Solution state for {solver} is: {solution_state}")

    print("Done")
