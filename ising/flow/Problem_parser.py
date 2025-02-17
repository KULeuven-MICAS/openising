import argparse
import sys

from ising.flow.MaxCut.MaxCut_benchmark import run_benchmark
from ising.flow.MaxCut.MaxCut_dummy import run_dummy
from ising.flow.TSP.TSP_benchmark import run_TSP_benchmark
from ising.flow.TSP.TSP_dummy import run_TSP_dummy

from ising.utils.flow import compute_list_from_arg

parser = argparse.ArgumentParser()
parser.add_argument("-problem", help="Which problem to solve", default="MaxCut")
parser.add_argument(
    "--N_list", help="tuple containing min and max problem size", default=None, nargs="+"
)
parser.add_argument("-benchmark", help="Which benchmark to run", default=None)
parser.add_argument("--iter_list", help="List of iterations", default=None, nargs="+")
parser.add_argument("-num_iter", help="The amount of iterations", default=None)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=3)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default=".png")

# TSP values
parser.add_argument("-weight_constant", help="Weight constant for TSP", default=1.0)

# Multiplicative parameters
parser.add_argument("-dtMult", help="time step for the Multiplicative solver", default=0.25)

# BRIM parameters
parser.add_argument("-dtBRIM", help="time step for the BRIM solver", default=0.25)
parser.add_argument("-C", help="capacitor parameter", default=1)
parser.add_argument("-stop_criterion", help="Stop criterion for change in voltages", default=1e-6)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=False, type=bool)

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

use_benchmark = False
use_dummy = False
if args.benchmark is not None and args.iter_list is not None:
    use_benchmark = True
    benchmark = args.benchmark
    iter_list = compute_list_from_arg(args.iter_list[0], 100)
elif args.N_list is not None and args.num_iter is not None:
    use_dummy = True
    N_list = compute_list_from_arg(args.N_list[0], 1 if problem == "TSP" else 10)
else:
    sys.exit("Cannot run solvers since no benchmark and N_list are given")

run_function = {
    ("MaxCut", True, False): (run_benchmark, ["benchmark", "iter_list", "solvers", "args"]),
    ("MaxCut", False, True): (run_dummy, ["N_list", "solvers", "args"]),
    ("TSP", True, False): (run_TSP_benchmark, ["benchmark", "iter_list", "solvers", "args"]),
    ("TSP", False, True): (run_TSP_dummy, ["N_list", "solvers", "args"]),
    # Add more problem types and conditions as needed
}

# Extract relevant arguments from args
kwargs = {}
kwargs.update({
    "benchmark": benchmark if use_benchmark else None,
    "iter_list": iter_list if use_benchmark else None,
    "N_list": N_list if use_dummy else None,
    "solvers": solvers,
    "args": args
})

key = (problem, use_benchmark, use_dummy)
if key in run_function:
    func, expected_args = run_function[key]
    filtered_kwargs = {key: kwargs[key] for key in expected_args if key in kwargs}
    func(**filtered_kwargs)
else:
    sys.exit("Invalid problem or configuration")
