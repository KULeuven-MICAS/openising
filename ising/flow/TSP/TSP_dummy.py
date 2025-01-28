import argparse
import os
import pathlib
import time
import numpy as np

from ising.generators.TSP import generate_random_TSP
from ising.flow.TSP.Calculate_TSP_energy import calculate_TSP_energy
from ising.utils.threading import make_solvers_thread, make_Gurobi_thread
from ising.utils.flow import make_directory, parse_hyperparameters, return_q, return_c0, return_rx

TOP = pathlib.Path(os.getenv("TOP"))


def run_TSP_dummy(N_list: list[int], solvers: list[str], args: argparse.Namespace):
    """Generates random TSP problems of sizes in the N_list and runs the specified solvers on them.

    Args:
        N_list (list[int]): the sizes of the TSP problems to generate
        solvers (list[str]): the solvers to run
        args (argparse.Namespace): the arguments parsed with ising/flow/Problem_parser.py
    """
    logpath = TOP / "ising/flow/TSP/logs"
    make_directory(logpath)

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
    for N in N_list:
        problems[N], graphs[N] = generate_random_TSP(N, seed)

    if use_gurobi:
        logfiles = {N: logpath / f"Gurobi_N{N}.log" for N in N_list}
        make_Gurobi_thread(models=problems, logfiles=logfiles)

        for N in N_list:
            calculate_TSP_energy([logfiles[N]], graphs[N], gurobi=True)


    for N in N_list:
        print(f"Running for {N} cities")
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
    print("Done")
