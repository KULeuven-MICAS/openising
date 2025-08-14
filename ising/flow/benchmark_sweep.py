import argparse
import logging
# import numpy as np
# import pathlib
import threading

from ising.flow import LOGGER, TOP
from ising.flow.MaxCut.MaxCut_benchmark import run_benchmark
from ising.flow.TSP.TSP_benchmark import run_TSP_benchmark
from ising.flow.TSP.ATSP_benchmark import run_ATSP_benchmark

from ising.utils.flow import go_over_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Go over all the benchmarks in a given directory. Possible benchmarks are Gset, TSP and ATSP."
    )
    parser.add_argument("-benchmark", help="The chosen benchmark", type=str, default="G")
    parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
    parser.add_argument("-nb_runs", help="Number of runs", default=5)
    parser.add_argument("-num_iter", help="The amount of iterations", default=None)
    parser.add_argument("-weight_constant", help="Weight constant for TSP", default=1.0)
    parser.add_argument("-fig_folder", help="Folder in which to save the figures", default="")
    parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
    parser.add_argument("-percentage", help="percentage of benchmarks to run", default=1.0)
    parser.add_argument('-part', help="Which part of the benchmark files to use. Starting from 0.", default=0)

    # Multiplicative parameters
    parser.add_argument("-dtMult", help="time step for the Multiplicative solver", default=0.01)
    parser.add_argument("-T_cont", help="Annealing temperature for continuous solvers", default=0.05)
    parser.add_argument("-T_final_cont", help="Final annealing temperature for continuous solvers", default=0.0005)
    parser.add_argument("-resistance", help="Resistance of the system", default=1.0)
    parser.add_argument("-flipping", help="Whether to use flipping", default=False)
    parser.add_argument("-flipping_freq", help="Frequency of flipping in Hz", default=10000)
    parser.add_argument("-flipping_prob", help="Probability of flipping", default=0.001799)
    parser.add_argument("-flipping_time", help="Time for the system to flip", default=5e-4)
    parser.add_argument("-mu_param", help="Mu parameter for the multiplicative solver", default=-3.55)

    # BRIM parameters
    parser.add_argument("-dtBRIM", help="time step for the BRIM solver", default=0.01)
    parser.add_argument("-capacitance", help="capacitor parameter", default=1)
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
    logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    benchmark = args.benchmark
    benchmark_list = go_over_benchmark(TOP / f"ising/benchmarks/{benchmark}", float(args.percentage), int(args.part))
    LOGGER.info(f"Benchmarks that will run are: {benchmark}")

    if benchmark == "G":
        func = run_benchmark
    elif benchmark == "ATSP":
        func = run_ATSP_benchmark
    else:
        func = run_TSP_benchmark

    # Create a thread for each benchmark
    if args.solvers == "all":
        solvers = ["BRIM", "SA", "bSB", "dSB", "SCA", "Multiplicative"]
    else:
        solvers = args.solvers[0].split()
    LOGGER.info(f"Solving with the following solvers: {solvers}")

    nb_runs = int(args.nb_runs)
    if nb_runs > 10:
        LOGGER.info("Number of runs is too high, setting it to 10")
        nb_runs = 10

    num_iter = [int(args.num_iter)]

    LOGGER.info("Threads are created")
    threads = []
    for benchmark in benchmark_list:
        args.fig_folder = benchmark
        args_thread = (benchmark, num_iter, solvers, args)
        threads.append(threading.Thread(target=func, args=args_thread, group=None))

    LOGGER.info("Threads are started")
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        # LOGGER.info(f"Thread {thread.name} finished")
    LOGGER.info("All threads finished")

if __name__ == "__main__":
    main()
