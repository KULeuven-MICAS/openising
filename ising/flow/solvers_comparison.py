import os
import pathlib
import numpy as np
import argparse
import openjij as oj
import random

from ising.generators.MaxCut import random_MaxCut

# import all solvers for comparison
from ising.solvers.BRIM import BRIM
from ising.solvers.DSA import DSASolver
from ising.solvers.SA import SASolver
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA

from ising.postprocessing.energy_plot import plot_energy_accuracy_check_mult_solvers


from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("--Nlist", help="tuple containing min and max problem size", default=(10, 100), nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)

# BRIM parameters
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-kmin", help="Minimum latch strength", default=0.01)
parser.add_argument("-kmax", help="Maximum latch strength", default=2.5)

# SA parameters
parser.add_argument("-T", help="Initial temperature", default=50.0)
parser.add_argument("-r_T", help="Temperature reduction rate", default=0.99)
parser.add_argument("-seed", help="Seed for random number generator", default=1)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=5.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)

print("parsing args")
args = parser.parse_args()
Nlist = tuple(args.Nlist)
nb_runs = int(args.nb_runs)
Nlist = np.linspace(Nlist[0], Nlist[1], nb_runs, dtype=int)
random.seed(int(args.seed))

num_iter = int(args.num_iter)

# BRIM parameters
tend = float(args.tend)
C = float(args.C)
G = float(args.G)
kmin = float(args.kmin)
kmax = float(args.kmax)

# SA parameters
T = float(args.T)
r_T = float(args.r_T)

# SCA parameters
q = float(args.q)
r_q = (float(args.q_final) / q) ** (1 / (num_iter + 1))

# SB parameters
dt = float(args.dt)


def at(t):
    return 1.0 / (dt * num_iter) * t


logfiles = dict()
logtop = TOP / "ising/flow/logs"
best_found = []
for N in Nlist:
    logfiles[N] = dict()
    logfiles[N]["BRIM"] = []
    logfiles[N]["DSA"] = []
    logfiles[N]["SA"] = []
    logfiles[N]["bSB"] = []
    logfiles[N]["dSB"] = []
    logfiles[N]["SCA"] = []

    print("Generating MaxCut problem of size", N)
    problem = random_MaxCut(N)

    print("Solving with OpenJij")
    mat = np.diag(problem.h) - triu_to_symm(problem.J)
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat)
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=nb_runs)
    best_found.append(response.first.energy)

    v = np.random.choice([-0.5, 0.5], (N,))
    x = np.random.uniform(-0.1, 0.1, (N,))
    y = np.zeros((N,))
    sigma = np.random.choice([-1.0, 1.0], (N,))

    c0 = 0.5 / (
        np.sqrt(problem.num_variables)
        * np.sqrt(np.sum(np.power(problem.J, 2)) / (problem.num_variables * (problem.num_variables - 1)))
    )

    for run in range(nb_runs):
        print("run ", run)
        print("Running BRIM")
        logfile = logtop / f"BRIM_N{N}_run{run}.log"
        BRIM().solve(
            model=problem,
            v=v,
            num_iterations=num_iter,
            dt=tend / num_iter,
            kmin=kmin,
            kmax=kmax,
            C=C,
            G=G,
            file=logfile,
        )
        logfiles[N]["BRIM"].append(logfile)

        print("running DSA")
        logfile = logtop / f"DSA_N{N}_run{run}.log"
        DSASolver().solve(
            model=problem,
            initial_state=sigma,
            num_iterations=num_iter,
            initial_temp=T,
            cooling_rate=r_T,
            file=logfile,
            seed=int(args.seed),
        )
        logfiles[N]["DSA"].append(logfile)

        print("running SA")
        logfile = logtop / f"SA_N{N}_run{run}.log"
        SASolver().solve(
            model=problem,
            initial_state=sigma,
            num_iterations=num_iter,
            initial_temp=T,
            cooling_rate=r_T,
            file=logfile,
            seed=int(args.seed),
        )
        logfiles[N]["SA"].append(logfile)

        print("running SB")
        logfile = logtop / f"bSB_N{N}_run{run}.log"
        ballisticSB().solve(model=problem, x=x, y=y, num_iterations=num_iter, at=at, a0=1.0, c0=c0, dt=dt, file=logfile)
        logfiles[N]["bSB"].append(logfile)

        logfile = logtop / f"dSB_N{N}_run{run}.log"
        discreteSB().solve(model=problem, x=x, y=y, num_iterations=num_iter, at=at, a0=1.0, c0=c0, dt=dt, file=logfile)
        logfiles[N]["dSB"].append(logfile)

        print("running SCA")
        logfile = logtop / f"SCA_N{N}_run{run}.log"
        SCA().solve(
            model=problem,
            sample=sigma,
            num_iterations=num_iter,
            T=T,
            r_t=r_T,
            q=q,
            r_q=r_q,
            seed=int(args.seed),
            file=logfile,
        )
        logfiles[N]["SCA"].append(logfile)
plot_energy_accuracy_check_mult_solvers(
    logfiles, best_found, save_folder=TOP / "ising/flow/plots", figName="Energy_accuracy_check_multiple_solvers.png"
)
