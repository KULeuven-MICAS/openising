import argparse
import sys
import logging

from ising.flow import LOGGER
from ising.flow.MaxCut.MaxCut_benchmark import run_benchmark
from ising.flow.MaxCut.MaxCut_dummy import run_dummy
from ising.flow.TSP.TSP_benchmark import run_TSP_benchmark
from ising.flow.TSP.TSP_dummy import run_TSP_dummy
from ising.flow.MIMO.MIMO_benchmarks import test_MIMO

from ising.utils.flow import compute_list_from_arg

parser = argparse.ArgumentParser()
parser.add_argument("-log_level", help="The level of logging output", default="INFO")

parser.add_argument("-problem", help="Which problem to solve", default="MaxCut")
parser.add_argument(
    "--N_list", help="tuple containing min and max problem size", default=None, nargs="+"
)
parser.add_argument("-benchmark", help="Which benchmark to run", default=None)
parser.add_argument("--iter_list", help="List of iterations", default=None, nargs="+")
parser.add_argument("-num_iter", help="The amount of iterations", default=None)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default=".png")

# TSP values
parser.add_argument("-weight_constant", help="Weight constant for TSP", default=1.0)

# MIMO SNR argument
parser.add_argument("--SNR", help='Signal to noise ratio', default=None, nargs="+")
parser.add_argument("-Nt", help="The amount of users", default=2)
parser.add_argument("-Nr", help="The amount of receivers", default=2)
parser.add_argument("-M", help="The QAM scheme", default=4)

# Multiplicative parameters
parser.add_argument("-dtMult", help="time step for the Multiplicative solver", default=0.01)
parser.add_argument("-T_cont", help="Annealing temperature for continuous solvers", default=0.05)
parser.add_argument("-T_final_cont", help="Final annealing temperature for continuous solvers", default=0.0005)
parser.add_argument("-coupling_annealing", help="Whether to anneal the coupling matrix", default=False)

# BRIM parameters
parser.add_argument("-dtBRIM", help="time step for the BRIM solver", default=0.01)
parser.add_argument("-C", help="capacitor parameter", default=1)
parser.add_argument("-stop_criterion", help="Stop criterion for change in voltages", default=1e-8)

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

# Set up logging
level = args.log_level
match level:
    case "INFO":
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)
    case "DEBUG":
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    case "WARNING":
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.WARNING)
    case "ERROR":
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.ERROR)
    case "CRITICAL":
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.CRITICAL)

problem = args.problem
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA", "Multiplicative"]
else:
    solvers = args.solvers[0].split()
LOGGER.info(f"Solving with the following solvers: {solvers}")

use_benchmark = False
use_dummy = False
use_MIMO = False
if args.benchmark is not None and args.iter_list is not None:
    use_benchmark = True
    benchmark = args.benchmark
    iter_list = compute_list_from_arg(args.iter_list[0], 100)
    LOGGER.debug("Running for the following iterations:" + str(iter_list))
elif args.N_list is not None and args.num_iter is not None:
    use_dummy = True
    N_list = compute_list_from_arg(args.N_list[0], 1 if problem == "TSP" else 10)
    LOGGER.debug("Running for the following problem sizes:" + str(N_list))
elif args.SNR is not None and args.num_iter is not None:
    use_MIMO = True
    SNR_list = compute_list_from_arg(args.SNR[0], 1)
    LOGGER.debug("Running for the following SNR values:" + str(SNR_list))
else:
    sys.exit("Problem is not implemented or some arguments are missing")

run_function = {
    ("MaxCut", True, False, False): (run_benchmark, ["benchmark", "iter_list", "solvers", "args"]),
    ("MaxCut", False, True, False): (run_dummy, ["N_list", "solvers", "args"]),
    ("TSP", True, False, False): (run_TSP_benchmark, ["benchmark", "iter_list", "solvers", "args"]),
    ("TSP", False, True, False): (run_TSP_dummy, ["N_list", "solvers", "args"]),
    ("MIMO", False, False, True): (test_MIMO, ["SNR_list", "solvers", "args"])
    # Add more problem types and conditions as needed
}

# Extract relevant arguments from args
kwargs = {}
kwargs.update({
    "benchmark": benchmark if use_benchmark else None,
    "iter_list": iter_list if use_benchmark else None,
    "N_list": N_list if use_dummy else None,
    "SNR_list": SNR_list if use_MIMO else None,
    "solvers": solvers,
    "args": args
})

key = (problem, use_benchmark, use_dummy, use_MIMO)
if key in run_function:
    LOGGER.info("Running...")
    func, expected_args = run_function[key]
    filtered_kwargs = {key: kwargs[key] for key in expected_args if key in kwargs}
    func(**filtered_kwargs)
    LOGGER.info("Done")
else:
    sys.exit("Invalid problem or configuration")
