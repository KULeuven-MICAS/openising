import numpy as np
import logging

from ising.flow import TOP, LOGGER
from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.utils.flow import make_directory
from ising.postprocessing.energy_plot import plot_energies_multiple

from ising.under_dev.SB_bit_accuracy.SB import ballisticSB

logtop = TOP / "ising/under_dev/SB_bit_accuracy/logs"
make_directory(logtop)
figtop = TOP / "ising/under_dev/SB_bit_accuracy/figures"
make_directory(figtop)
logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

def main():
    bit_width_list_y = [4, 8, 16]
    bit_width_x =  16
    benchmark = "G1"
    graph, best_found = G_parser(TOP / f"ising/benchmarks/G/{benchmark}.txt")
    LOGGER.info(f"best found: {-best_found}")

    model = MaxCut(graph)
    a0 = 1.0
    c0 = 0.7/np.sqrt(model.num_variables)
    dt = 0.25
    num_iter = 5000

    for bit_width in bit_width_list_y:
        logfile = logtop / f"SB_{benchmark}_bitwidth_{bit_width}_x_{bit_width_x}.log"
        LOGGER.info(f"bit_width y: {bit_width}")

        state, energy = ballisticSB().solve(model, num_iter, c0, dt, a0, logfile, bit_width_x, bit_width)
        LOGGER.info(f"Solved with energy: {energy}")

def test():
    benchmark = "G1"
    graph, best_found = G_parser(TOP / f"ising/benchmarks/G/{benchmark}.txt")
    LOGGER.info(f"best found: {-best_found}")

    model = MaxCut(graph)
    a0 = 1.0
    c0 = 0.7/np.sqrt(model.num_variables)
    dt = 0.25
    num_iter = 5000
    logfile = logtop / f"SB_test_{benchmark}_bitwidth_16.log"

    state, energy = ballisticSB().solve(model, num_iter, c0, dt, a0, logfile, 16)
    LOGGER.info(f"Solved with energy: {energy} and state: {state}")

def plot_logs():
    logfiles = [logtop / f"SB_G1_bitwidth_{y}_x_{bit_width}.log" for y in [4, 8, 16] for bit_width in [4, 8, 16]]
    plot_energies_multiple(logfiles, figName="SB_bit_accuray_G1_diff_quantization.png", best_found=-11624, save_folder=figtop, percentage=0.4)

if __name__ == "__main__":
    # main()
    # test()
    plot_logs()
