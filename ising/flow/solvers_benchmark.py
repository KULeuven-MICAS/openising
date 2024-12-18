import numpy as np
import os
import argparse
import openjij as oj
import matplotlib.pyplot as plt
import pathlib

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.solvers.BRIM import BRIM
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.DSA import DSASolver

TOP = pathlib.Path(os.getenv("TOP"))

parser = argparse.ArgumentParser()
parser.add_argument()
parser.add_argument("-benchmark", help="Name of the benchmark to run", default="K2000")
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-figName", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")

# BRIM parameters
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-kmin", help="Minimum latch strength", default=0.01)
parser.add_argument("-kmax", help="Maximum latch strength", default=3)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=False)

# SA parameters
parser.add_argument("-T", help="Initial temperature", default=50.0)
parser.add_argument("-Tfin", help="Final temperature of the annealing process", default=0.05)
parser.add_argument("-seed", help="Seed for random number generator", default=1)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=5.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)