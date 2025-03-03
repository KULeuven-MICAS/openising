import pathlib
import os
import argparse

from ising.postprocessing.plot_solutions import plot_state
from ising.postprocessing.MIMO_plot import plot_error_SNR
from ising.utils.flow import make_directory, compute_list_from_arg
from ising.utils.HDF5Logger import return_metadata

TOP = pathlib.Path(os.getenv("TOP"))

parser = argparse.ArgumentParser()
parser.add_argument("--SNR", help="Range of Signal to Noise ratios", default=None, nargs="+")
parser.add_argument("--solvers", help="Which solvers to plot", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="MIMO_test.png")
parser.add_argument("-M", help="QAM modulation scheme", default=4)
args = parser.parse_args()

figtop = TOP / "ising/flow/MIMO/plots" / args.fig_folder
make_directory(figtop)

if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM", "Multiplicative"]
else:
    solvers = args.solvers[0].split()
nb_runs = int(args.nb_runs)

SNR_list = compute_list_from_arg(args.SNR[0])
logfiles = {solver: [] for solver in solvers}
logtop = TOP / "ising/flow/MIMO/logs"

actual_solutions = {SNR: dict() for SNR in SNR_list}
for solver in solvers:
    for SNR in SNR_list:
        logfiles[solver] = [logtop / f"{solver}_SNR{SNR}_run{run}.log" for SNR in SNR_list for run in range(nb_runs)]
        for run in range(nb_runs):
            solution_file = logtop / f"actual_solution_SNR{SNR}_run{run}.log"
            xtilde = return_metadata(solution_file, "x")
            actual_solutions[SNR][run] = xtilde
        plot_state(
            solver,
            logtop / f"{solver}_SNR{SNR}_run{run}.log",
            figname=f"{solver}_SNR{SNR}_{args.fig_name}",
            figtop=figtop,
            )

plot_error_SNR(logfiles, int(args.M), actual_solutions, save_folder=figtop, figname="error_SNR_" + args.fig_name)
