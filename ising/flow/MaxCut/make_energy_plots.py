import numpy as np
import argparse
import sys


from ising.flow import LOGGER, TOP
from ising.utils.parser import get_optim_value
from ising.postprocessing.energy_plot import (
    plot_energy_dist_multiple_solvers,
    plot_energies_multiple,
)
from ising.postprocessing.summarize_energies import summary_energies
from ising.postprocessing.plot_solutions import plot_state
from ising.utils.flow import compute_list_from_arg
from ising.utils.helper_functions import make_directory
from ising.utils.HDF5Logger import get_Gurobi_data

# Defining all arguments
parser = argparse.ArgumentParser()
parser.add_argument("--solvers", help="Which solvers to gather data from", default="all", nargs="+")
parser.add_argument("-benchmark", help="Name of the banchmark that ran", default=None)
parser.add_argument("--N_list", help="Tuple containing min and max problem size", default=None, nargs="+")
parser.add_argument("--iter_list", help="Number of iterations", default=None, nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-use_gurobi", help="whether Gurobi was used", default=False)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="best_energy")

# Parsing the arguments
args = parser.parse_args()

# Setting the solvers and the amount of runs
if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM", "Multiplicative"]
else:
    solvers = args.solvers[0].split()

LOGGER.info("Plotting for solvers: " + str(solvers))
nb_runs = int(args.nb_runs)

# Defining the top paths and list for the logfiles
logfiles = []
logtop = TOP / "ising/outputs/Maxcut/logs"
figtop = TOP / "ising/flow/MaxCut/plots" / args.fig_folder
make_directory(figtop)
fig_name = str(args.fig_name)

use_gurobi = bool(int(args.use_gurobi))

if args.benchmark is not None:
    LOGGER.info("Benchmark logs are plotted")
    # Benchmark is given and should be plotted
    benchmark = str(args.benchmark)

    # Check if num_iter is given
    if args.iter_list is None:
        sys.exit("No iteration range is specified while benchmark is given")
    num_iter = args.iter_list[0]
    iter_list = compute_list_from_arg(num_iter, 100)

    # Get the best found of the benchmark
    best_found = get_optim_value(
        benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt", optim_file=TOP / "ising/benchmarks/G/optimal_energy.txt"
    )

    # Go over all solvers and generate the logfiles
    for num_iter in iter_list:
        new_logfiles = [
            logtop / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
            for solver in solvers
            for run in range(nb_runs)
        ]
        for solver in solvers:
            for run in range(nb_runs):
                if run == nb_runs - 1:
                    plot_state(
                        solver,
                        logtop / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log",
                        f"{solver}_benchmark{benchmark}_state_iter{num_iter}",
                        figtop=figtop,
                    )

        plot_energies_multiple(
            logfiles=new_logfiles,
            figName=f"{benchmark}_nb_iter{num_iter}_{fig_name}",
            best_found=best_found,
            save_folder=figtop,
        )
        logfiles += new_logfiles

    make_directory(figtop / "summary_energies")
    summary_energies(logfiles, figtop / "summary_energies")


elif args.N_list is not None:
    LOGGER.info("Problem size logs are plotted")
    # List of problem sizes is given
    N_list = args.N_list[0].split()
    N_list = np.array(range(int(N_list[0]), int(N_list[1]), 10))
    best_found = []

    # Generate all the logfiles
    for N in N_list:
        new_logfiles = []
        for solver in solvers:
            for run in range(nb_runs):
                logfile = logtop / f"{solver}_N{N}_run{run}.log"
                new_logfiles.append(logfile)
            if run == nb_runs - 1:
                plot_state(solver, logfile, f"{solver}_N{N}", figtop)
        if use_gurobi:
            best_found.append(logtop / f"Gurobi_N{N}.log")
        plot_energies_multiple(
            logfiles=new_logfiles,
            figName=f"N{N}_{fig_name}",
            best_found=get_Gurobi_data([best_found[-1]]) if use_gurobi else None,
            save_folder=figtop,
        )
        logfiles += new_logfiles

    # Make sure that best_found is None if Gurobi was not used
    if len(best_found) == 0:
        best_found = None
    else:
        best_found = np.array(get_Gurobi_data(best_found))

    plot_energy_dist_multiple_solvers(
        logfiles,
        best_found=best_found,
        best_Gurobi=None,
        xlabel="problem_size",
        save_folder=figtop,
        figName=f"size_comparison_{fig_name}",
    )

else:
    # No benchmark or problem size range is given => exit
    sys.exit("No benchmark or problem size range is specified")

LOGGER.info("figures plotted succesfully")
