import argparse
import sys

from ising.flow.MaxCut.MaxCut_benchmark import run_benchmark
from ising.flow.MaxCut.MaxCut_dummy import run_dummy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--N_list", help="tuple containing min and max problem size", default=None, nargs="+"
)
parser.add_argument("-benchmark", help="Which benchmark to run", default=None)
parser.add_argument("--iter_list", help="List of iterations", default=None, nargs="+")
parser.add_argument("-num_iter", help="The amount of iterations", default=None)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=3)
parser.add_argument("-fig_folder", help="Folder inwhich to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")

parser.add_argument("-dummy", help="Whether to run the dummy function", default=False)

# Multiplicative parameters
parser.add_argument("-dtMult", help="time step for the Multiplicative solver", default=0.25)

# BRIM parameters
# parser.add_argument("-dtBRIM", help="time step for the BRIM solver", default=0.25)
parser.add_argument("-C", help="capacitor parameter", default=1)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-k_min", help="Minimum latch strength", default=0.01)
parser.add_argument("-k_max", help="Maximum latch strength", default=2.5)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=False)
parser.add_argument("-latch", help="whether to turn on the latches", default=False)

# SA parameters
parser.add_argument("-T", help="Initial temperature", default=50.0)
parser.add_argument("-T_final", help="Final temperature of the annealing process", default=0.05)
parser.add_argument("-seed", help="Seed for random number generator", default=0)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=0.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dtSB", help="Time step for simulated bifurcation", default=0.25)
parser.add_argument("-a0", help="Parameter a0 of SB", default=1.0)
parser.add_argument("-c0", help="Parameter c0 of SB", default=0.0)

args = parser.parse_args()
problem = args.problem
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA", "Multiplicative"]
else:
    solvers = args.solvers[0].split()
print(f"Solving with the following solvers: {solvers}")

if problem == "MaxCut":
    if args.benchmark is not None:
        benchmark = args.benchmark
        if args.iter_list is None:
            sys.exit("No range of iterations is given")
        run_benchmark(benchmark, args.iter_list[0], args)
    elif args.N_list is not None:
        N_list = args.N_list[0]
        if args.num_iter is None:
            sys.exit("No number of iterations is given")
        run_dummy(N_list, solvers, args)
    else:
        sys.exit("Cannot run solvers since no benchmark and N_list are given")
if problem == "TSP":
    pass
