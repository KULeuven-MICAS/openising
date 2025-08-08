import argparse
import sys
import numpy as np

from ising.flow import TOP, LOGGER
from ising.utils.parser import get_optim_value
from ising.postprocessing.energy_plot import (
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
parser.add_argument("-percentage", help="Amount of percentage to plot of energies", default=1.0)

# Parsing the arguments
args = parser.parse_args()

use_benchmark = False
use_dummy = False
if args.benchmark is not None and args.iter_list is not None:
    use_benchmark = True
    benchmark = args.benchmark
    iter_list = compute_list_from_arg(args.iter_list[0], 100)
    LOGGER.debug(f"Using benchmark {benchmark} with iterations {iter_list}")
elif args.N_list is not None:
    use_dummy = True
    N_list = compute_list_from_arg(args.N_list[0], 1)
    LOGGER.debug(f"Using dummy with sizes {N_list}")
else:
    sys.exit("Cannot run solvers since no benchmark and N_list are given")

# Setting the solvers and the amount of runs
if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM", "Multiplicative"]
else:
    solvers = args.solvers[0].split()
nb_runs = int(args.nb_runs)

# Defining the top paths and list for the logfiles
logfiles = []
logtop = TOP / "ising/outputs/TSP/logs"
figtop = TOP / "ising/flow/TSP/plots_TSP" / args.fig_folder
make_directory(figtop)
LOGGER.debug(f"Saving figures in {figtop}")
fig_name = str(args.fig_name)

if args.benchmark is not None:
    LOGGER.info("Benchmark logs are plotted")
    # Benchmark is given and should be plotted
    benchmark = str(args.benchmark)

    # Get the best found of the benchmark
    best_found = get_optim_value(
        benchmark=TOP / f"ising/benchmarks/TSP/{benchmark}.tsp",
        optim_file=TOP / "ising/benchmarks/TSP/optimal_energy.txt",
    )

    # Go over all solvers and generate the logfiles
    for num_iter in iter_list:
        new_logfiles = [
            logtop / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
            for solver in solvers
            for run in range(nb_runs)
        ]
        for solver in solvers:
            run = nb_runs - 1
            plot_state(
                solver,
                logtop / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log",
                f"{solver}_benchmark{benchmark}_state_iter{num_iter}",
                figtop=figtop,
            )

        plot_energies_multiple(
            logfiles=new_logfiles,
            figName=f"{benchmark}_nb_iter{num_iter}_{fig_name}",
            y_data="energy",
            best_found=best_found,
            save_folder=figtop,
            percentage=float(args.percentage),
        )
        logfiles += new_logfiles
    if bool(args.use_gurobi):
        best_found_gurobi = [logtop / f"Gurobi_{benchmark}.log"]
        best_found_gurobi = get_Gurobi_data(np.array(best_found_gurobi), metadata="solution_TSP_energy") * np.ones(
            len(iter_list)
        )
    else:
        best_found_gurobi = None

    if best_found is not None:
        best_found = np.ones((len(iter_list),)) * best_found

    make_directory(figtop / "energy_summary")
    summary_energies(logfiles, figtop / "energy_summary")

elif args.N_list is not None:
    LOGGER.info("Problem size logs are plotted")
    # List of problem sizes is given
    best_found_gurobi = []

    # Generate all the logfiles
    logfiles = [logtop / f"{solver}_N{N}_run{run}.log" for N in N_list for solver in solvers for run in range(nb_runs)]
    if bool(args.use_gurobi):
        best_found_gurobi = [logtop / f"Gurobi_N{N}.log" for N in N_list]
    run = nb_runs - 1
    for N in N_list:
        for solver in solvers:
            plot_state(solver, logtop / f"{solver}_N{N}_run{run}.log", f"{solver}_N{N}_{fig_name}", figtop)

    # Make sure that best_found is None if Gurobi was not used
    if len(best_found_gurobi) == 0:
        best_found_gurobi = None
    else:
        best_found_gurobi = np.array(get_Gurobi_data(best_found_gurobi, metadata="solution_TSP_energy"))
    best_found = None

else:
    # No benchmark or problem size range is given => exit
    sys.exit("No benchmark or problem size range is specified")

LOGGER.info("figures plotted succesfully")
