import argparse
import logging

from ising.flow import LOGGER, TOP
from ising.utils.flow import go_over_benchmark
from ising.postprocessing.plot_all_benchmarks import plot_energy_distribution, plot_energy_average
from ising.utils.flow import make_directory

parser = argparse.ArgumentParser(
    description="Plot the solution energy of all the benchmarks in a given set. "
    "Possible benchmarks are Gset, TSP and ATSP."
)
parser.add_argument("-benchmark", help="The chosen benchmark", type=str, default="G")
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=5)
parser.add_argument("-num_iter", help="The amount of iterations", default=None)
parser.add_argument("-fig_folder", help="Folder of the figure to save", default=".")
parser.add_argument("-fig_name", help="Name of the figure to save", default="benchmark_sweep.png")
parser.add_argument("-percentage", help="percentage of benchmarks to run", default=1.0)


args = parser.parse_args()
logging.basicConfig(format="%(levelname)s:%(message)s", force=True, level=logging.INFO)

benchmark = args.benchmark
LOGGER.info(f"Benchmarks that will be plotted are: {benchmark}")
benchmark_list = go_over_benchmark(TOP / f"ising/benchmarks/{benchmark}", float(args.percentage))

# Create a thread for each benchmark
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA", "Multiplicative"]
else:
    solvers = args.solvers[0].split()

num_iter = int(args.num_iter)
nb_runs = int(args.nb_runs)

top = TOP / "ising/flow"
if benchmark == "G":
    figtop = top / "MaxCut/plots"
    logtop = top / "MaxCut/logs"
elif benchmark == "ATSP":
    logtop = top / "TSP/logs_ATSP"
    figtop = top / "TSP/plots_ATSP"
else:
    logtop = top / "TSP/logs_TSP"
    figtop = top / "TSP/plots_TSP"
make_directory(figtop / args.fig_folder)

logfiles = [
    logtop / f"{solver}_{bench}_nbiter{num_iter}_run{run}.log"
    for solver in solvers
    for bench in benchmark_list
    for run in range(nb_runs)
]

LOGGER.info("plotting data of the logfiles")

plot_energy_distribution(
    logfiles,
    benchmark,
    "distribution_" + args.fig_name,
    save_dir=figtop / args.fig_folder,
    percentage=float(args.percentage),
)
plot_energy_average(
    logfiles,
    benchmark,
    "average_energies_" + args.fig_name,
    save_dir=figtop / args.fig_folder,
    percentage=float(args.percentage),
)
