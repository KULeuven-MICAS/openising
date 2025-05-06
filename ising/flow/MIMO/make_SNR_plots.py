import argparse

from ising.flow import LOGGER, TOP
from ising.postprocessing.plot_solutions import plot_state
from ising.postprocessing.energy_plot import plot_energies_multiple
from ising.postprocessing.MIMO_plot import plot_error_SNR
from ising.utils.flow import make_directory, compute_list_from_arg

parser = argparse.ArgumentParser()
parser.add_argument("--SNR", help="Range of Signal to Noise ratios", default=None, nargs="+")
parser.add_argument("--solvers", help="Which solvers to plot", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="MIMO_test.png")
parser.add_argument("-M", help="QAM modulation scheme", default=4)
args = parser.parse_args()

figtop = TOP / "ising/flow/MIMO/plots" / args.fig_folder
make_directory(figtop)
LOGGER.debug(f"Fig folder: {figtop}")
logtop = TOP / "ising/flow/MIMO/logs"

if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM", "Multiplicative"]
else:
    solvers = args.solvers[0].split()
nb_runs = int(args.nb_runs)
SNR_list = compute_list_from_arg(args.SNR[0])

use_gurobi = bool(int(args.use_gurobi))

if use_gurobi:
    gurobi_files = [logtop / f"Gurobi_SNR{SNR}_run{run}.log" for SNR in SNR_list for run in range(nb_runs)]
else:
    gurobi_files = None

logfiles = []
for SNR in SNR_list:
    current_logfiles = [logtop / f'{solver}_SNR{SNR}_run{run}.log' for solver in solvers for run in range(nb_runs)]
    logfiles += current_logfiles
    plot_energies_multiple(current_logfiles, figName=f"SNR{SNR}_energies_{args.fig_name}", save_folder=figtop)
    for solver in solvers:
        plot_state(
            solver,
            logtop / f"{solver}_SNR{SNR}_run{0}.log",
            figname=f"{solver}_SNR{SNR}_{args.fig_name}",
            figtop=figtop,
            )

plot_error_SNR(logfiles, gurobi_files, save_folder=figtop, figname="error_SNR_" + args.fig_name)
LOGGER.info("Done plotting figures")
