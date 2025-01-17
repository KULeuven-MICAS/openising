import numpy as np
import argparse
import sys
import os
import pathlib

from ising.benchmarks.parsers.G import get_optim_value
from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.postprocessing.plot_solutions import plot_state_continuous, plot_state_discrete
from ising.utils.flow import make_directory

TOP = pathlib.Path(os.getenv("TOP"))

# Defining all arguments
parser = argparse.ArgumentParser()
parser.add_argument("--solvers", help="Which solvers to gather data from", default="all", nargs="+")
parser.add_argument("-benchmark", help="Name of the banchmark that ran", default=None)
parser.add_argument("--N_list", help="Tuple containing min and max problem size", default=None, narg="+")
parser.add_argument("--num_iter", help="Range of number of iterations", default=None, nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-use_gurobi", help="whether Gurobi was used", default=False)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")

# Parsing the arguments
args = parser.parse_args()

# Setting the solvers and the amount of runs
if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM"]
else:
    solvers = args.solvers
nb_runs = int(args.nb_runs)

# Defining the top paths and list for the logfiles
logfiles = []
logtop = TOP / "ising/flow/MaxCut/logs"
figtop = TOP / "ising/flow/MaxCut/plots" / args.fig_folder
make_directory(figtop)

# Small function to plot the states
def plot_state(solver, logfile, figname):
    if solver in ["BRIM", "bSB", "dSB"]:
        plot_state_continuous(logfile=logfile, figname=figname, save_folder=figtop)
    else:
        plot_state_discrete(logfile=logfile, figname=figname, save_folder=figtop)


if args.benchmark is not None:
    # Benchmark is given and should be plotted
    benchmark = str(args.benchmark)

    # Check if num_iter is given
    if args.num_iter is None:
        sys.exit("No iteration range is specified while benchmark is given")
    num_iter = tuple(args.num_iter)
    iter_list = np.linspace(int(num_iter[0]), int(num_iter[1]), nb_runs, dtype=int)

    # Get the best found of the benchmark
    best_found = get_optim_value(benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt")
    if best_found is not None:
        best_found = [-best_found]*len(num_iter)

    # Go over all solvers and generate the logfiles
    for nb_iter in iter_list:
        for solver in solvers:
            for run in range(nb_runs):
                logfile = logtop / f"{solver}_{benchmark}_nbiter{nb_iter}_run{run}.log"
                logfiles.append(logfile)
                if run == nb_runs - 1:
                    plot_state(solver, logfile, f"{solver}_benchmark{benchmark}_state_iter{nb_iter}.png")

elif args.N_list is not None:
    # List of problem sizes is given
    N_list = tuple(args.N_list)
    N_list = np.linspace(N_list[0], N_list[1], nb_runs, dtype=int)
    best_found = []

    # Generate all the logfiles
    for N in N_list:
        for solver in solvers:
            for run in range(nb_runs):
                logfile = logtop / f"{solver}_N{N}_run{run}.log"
                logfiles.append(logfile)
            if run == nb_runs - 1:
                plot_state(solver, logfile, f"{solver}_N{N}.png", figtop)
        if bool(args.use_gurobi):
            best_found.append(logtop / f"Gurobi_N{N}.log")

    # Make sure that best_found is None if Gurobi was not used
    if len(best_found) == 0:
        best_found = None

else:
    # No benchmark or problem size range is given => exit
    sys.exit("No benchmark or problem size range is specified")

# Plot the energy distribution with the generated logfiles
plot_energy_dist_multiple_solvers(
    logfiles,
    best_found=best_found,
    best_Gurobi=bool(args.use_gurobi),
    xlabel="num_iterations" if args.benchmark is not None else "problem_size",
    save_folder=figtop,
    fig_name=f"energy_dist_{benchmark}.png" if args.benchmark is not None else "energy_dist_N.png",
)
