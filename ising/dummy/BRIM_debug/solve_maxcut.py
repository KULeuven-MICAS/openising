import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from ising.utils.flow import make_directory, return_c0
from ising.generators.MaxCut import MaxCut, random_MaxCut
from ising.benchmarks.parsers import G_parser
from ising.model.ising import IsingModel
from ising.dummy.BRIM_debug.BRIM_afoakwa import BRIMafoakwa
from ising.dummy.BRIM_debug.BRIM_current import BRIMcurrent
from ising.dummy.BRIM_debug.BRIM_roychowdhury import BRIMroychowdhury
from ising.dummy.BRIM_debug.Multiplicative import Multiplicative
from ising.solvers.exhaustive import ExhaustiveSolver

from ising.solvers.SB import ballisticSB

from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers, plot_relative_error, plot_energies_multiple
from ising.utils.HDF5Logger import return_data, return_metadata

TOP = pathlib.Path(os.getenv("TOP"))
logtop = TOP / "ising/dummy/BRIM_debug/logs"
make_directory(logtop)
figtop = TOP / "ising/dummy/BRIM_debug/figures"
make_directory(figtop)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def test_afoakwa():
    J = np.array([[0, -1, -1/2, 1],
                  [0, 0, 0, -3],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    h = np.zeros((4,))
    model = IsingModel(J, h)
    initial_state = np.array([1, 1, 1, -1])
    dt = 0.01
    num_iterations = 200
    C=0.05

    logfile = logtop / "BRIM_afoakwa_test.log"
    state, energy = BRIMafoakwa(model, initial_state, num_iterations, dt, C, logfile)
    print(f"Final energy: {energy} and state: {state}")
    plot_state_continuous(logfile, figname="BRIM_afoakwa_test.png")

def test_afoakwa2():
    N_list         = range(16, 28, 4)
    dt             = 0.01
    num_iterations = 5000
    C              = 1.
    seed           = 1
    Temp           = 1.
    cooling_rate   = (0.005 / Temp) ** (1 / (num_iterations + 1))
    stop_criterion = 1e-10
    np.random.seed(seed)

    nb_runs = 20
    logfiles = []
    best = []
    for N in N_list:
        current_logfiles = []
        problem = random_MaxCut(N, seed)
        
        c0 = return_c0(problem)
        # Solve with exhaustive solver
        _, optim = ExhaustiveSolver().solve(problem)
        best.append(optim)
        logger.info("Best found for %s nodes is: %s", N, optim)

        for run in range(nb_runs):
            initial_state = np.random.choice([-1, 1], N)
            # Solve with baseline ballistic SB
            logfile_SB = logtop / f"bSB_N{N}_run{run}.log"
            logfiles.append(logfile_SB)
            current_logfiles.append(logfile_SB)

            ballisticSB().solve(problem, 
                                initial_state, 
                                num_iterations, 
                                c0, 
                                dt, 
                                file=logfile_SB, 
                                bit_width=32)
            logger.info(f"run {run} of bSB done")

            # Solve with Afoakwa
            logfile_BRIM = logtop / f"BRIM_afoakwa_N{N}_run{run}.log"
            BRIMafoakwa(problem, 
                        initial_state, 
                        num_iterations, 
                        dt, 
                        C, 
                        logfile_BRIM, 
                        stop_criterion, 
                        Temp, 
                        cooling_rate, 
                        seed)
            logfiles.append(logfile_BRIM)
            current_logfiles.append(logfile_BRIM)
            logger.info(f"run {run} of BRIM done")

        plot_state_continuous(logfile_SB, figname=f"bSB_N{N}_test.png")
        plot_state_continuous(logfile_BRIM, figname=f"BRIM_afoakwa_N{N}_test.png")
        plot_energies_multiple(current_logfiles, figName=f"N{N}_energy_test.png", save_folder=figtop, best_found=optim)

    best = np.array(best)
    plot_energy_dist_multiple_solvers(logfiles, xlabel="problem_size", fig_name="small_energy_all_test.png", save_folder=figtop, best_found=best)
    plot_relative_error(logfiles, best, x_label="problem_size", fig_name="relative_error_test.png", save_folder=figtop)


def test_small():
    N_list = range(16, 33, 4)
    dt = 0.01
    C = 1.
    num_iterations = 10000
    seed = 1
    np.random.seed(seed)
    max_change = 1e-10
    Temp = 1.
    cooling_rate = (0.005 / Temp) ** (1 / (num_iterations + 1))
    
    logfiles = []
    best = []
    for N in N_list:
        current_logfiles = []
        problem = random_MaxCut(N, seed)

        _, optim = ExhaustiveSolver().solve(model=problem)
        best.append(optim)
        logger.info(f"Optimal energy for {N=} is {optim}")

        initial_state = np.random.choice([-1, 1], N)

        logfile = logtop / f"bSB_N{N}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        _, energy = ballisticSB().solve(problem, initial_state, num_iterations, 1., dt, 1.,  logfile)
        plot_state_continuous(logfile, figname=f"bSB_N{N}.png")
        logger.info("bSB done with energy: %s", energy)

        logfile = logtop / f"Multiplicative_N{N}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        _, energy = Multiplicative(problem, initial_state, num_iterations, dt, C, logfile, seed=seed)
        plot_state_continuous(logfile, figname=f"Multiplicative_N{N}.png")
        logger.info("Multiplicative done with energy: %s", energy)

        logfile = logtop / f"BRIM_current_N{N}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        _, energy = BRIMcurrent(problem, initial_state, num_iterations, dt, C, logfile, True, Temp, cooling_rate, seed)
        plot_state_continuous(logfile, figname=f"BRIM_current_N{N}.png")
        logger.info("BRIM current done with energy: %s", energy)

        logfile = logtop / f"BRIM_roychowdhury_N{N}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        _, energy = BRIMroychowdhury(problem, initial_state, num_iterations, dt, C, logfile, max_change, seed)
        plot_state_continuous(logfile, figname=f"BRIM_roychowdhury_N{N}.png")
        logger.info("BRIM roychowdhury done with energy: %s", energy)

        logfile = logtop / f"BRIM_afoakwa_N{N}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        _, energy = BRIMafoakwa(problem, initial_state, num_iterations, dt, C, logfile, max_change, Temp, cooling_rate, seed)
        plot_state_continuous(logfile, figname=f"BRIM_afoakwa_N{N}.png")
        logger.info("BRIM afoakwa done with energy: %s", energy)

        plot_energies_multiple(current_logfiles, figName=f"N{N}_energy.png", save_folder=figtop, best_found=optim)
    
    best = np.array(best)
    logger.debug("best found: %s", best)
    plot_energy_dist_multiple_solvers(logfiles, xlabel="problem_size", fig_name="small_energy_all.png", save_folder=figtop, best_found=best)
    plot_relative_error(logfiles, best, x_label="problem_size", fig_name="small_relative_error.png", save_folder=figtop)

def test1():
    dt = 0.05
    num_iterations = 1000
    C = 1.0
    seed = 1
    Temp = 50
    cooling_rate = (0.5 / Temp) ** (1 / (num_iterations + 1))
    np.random.seed(seed)

    N_list = [150, 200, 250, 300]
    logfiles = []
    for N in N_list:
        logger.info("Solving problem of size %s", N)
        problem = random_MaxCut(N, seed)
        initial_state = np.random.choice([-1, 1], N)


        logfile = logtop / f"bSB_N{N}.log"
        logfiles.append(logfile)
        ballisticSB().solve(problem, initial_state, num_iterations, c0=1., dtSB=0.25, file=logfile, a0=1.)
        plot_state_continuous(logfile, figname=f"bSB_N{N}.png")
        logger.info("bSB done")

        logfile = logtop / f"BRIM_current_N{N}.log"
        logfiles.append(logfile)
        BRIMcurrent(problem, initial_state, num_iterations, dt, C, logfile, random_flip=True, initial_temp=Temp, cooling_rate=cooling_rate, seed=seed)
        plot_state_continuous(logfile, figname=f"BRIM_current_N{N}.png")
        logger.info("BRIM current done")

        logfile = logtop / f"BRIM_roychowdhury_N{N}.log"
        logfiles.append(logfile)
        BRIMroychowdhury(problem, initial_state, num_iterations, dt, C, logfile, seed=seed)
        plot_state_continuous(logfile, figname=f"BRIM_roychowdhury_N{N}.png")
        logger.info("BRIM roychowdhury done")

        logfile = logtop / f"BRIM_afoakwa_N{N}.log"
        logfiles.append(logfile)
        BRIMafoakwa(problem, initial_state, num_iterations, dt, C, logfile, initial_temp=Temp, cooling_rate=cooling_rate, seed=seed)
        plot_state_continuous(logfile, figname=f"BRIM_afoakwa_N{N}.png")
        logger.info("BRIM afoakwa done")

    plot_energy_dist_multiple_solvers(logfiles, xlabel="problem_size", fig_name="energy_all.png", save_folder=figtop)

def test2():
    dt = 0.01
    bench = "G6"
    benchmark, best_found = G_parser(TOP / f"ising/benchmarks/G/{bench}.txt")
    problem = MaxCut(benchmark)
    C = 1
    seed = 1
    Temp = 25
    np.random.seed(seed)
    max_change = 1e-10

    c0 = return_c0(problem)
    iter_list = range(1000, 5001, 500)
    logfiles = []
    best = []
    for num_iterations in iter_list:
        logger.info("Iteration length %s", num_iterations)
        current_logfiles = []

        best.append(best_found)
        cooling_rate = (0.05 / Temp) ** (1 / (num_iterations + 1))
        N = problem.num_variables
        initial_state = np.random.choice([-1, 1], N)

        logfile = logtop / f"bSB_{bench}_iter{num_iterations}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        ballisticSB().solve(problem, initial_state, num_iterations, c0=c0, dtSB=0.25, file=logfile, a0=1.)
        plot_state_continuous(logfile, figname=f"bSB_{bench}_iter{num_iterations}.png")
        logger.info("bSB done")

        logfile = logtop / f"Multiplicative_{bench}_iter{num_iterations}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        Multiplicative(problem, 
                       initial_state, 
                       num_iterations, 
                       dt, 
                       C, 
                       logfile, 
                       random_flip=True, 
                       initial_temp=Temp, 
                       cooling_rate=cooling_rate, 
                       seed=seed, 
                       stop_criterion=max_change)
        plot_state_continuous(logfile, figname=f"Multiplicative_{bench}_iter{num_iterations}.png")
        logger.info("Multiplicative done")

        logfile = logtop / f"BRIM_current_{bench}_iter{num_iterations}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        BRIMcurrent(problem, 
                    initial_state, 
                    num_iterations, 
                    dt, 
                    C,
                    logfile, 
                    random_flip=True, 
                    initial_temp=Temp, 
                    cooling_rate=cooling_rate, 
                    seed=seed, 
                    stop_criterion=max_change)
        plot_state_continuous(logfile, figname=f"BRIM_current_{bench}_iter{num_iterations}.png")
        logger.info("BRIM current done")

        logfile = logtop / f"BRIM_roychowdhury_{bench}_iter{num_iterations}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        BRIMroychowdhury(problem, initial_state, num_iterations, dt, C, logfile, seed=seed, stop_criterion=max_change)
        plot_state_continuous(logfile, figname=f"BRIM_roychowdhury_{bench}_iter{num_iterations}.png")
        logger.info("BRIM roychowdhury done")

        logfile = logtop / f"BRIM_afoakwa_{bench}_iter{num_iterations}.log"
        logfiles.append(logfile)
        current_logfiles.append(logfile)
        BRIMafoakwa(problem, initial_state, num_iterations, dt, C, logfile, initial_temp=Temp, cooling_rate=cooling_rate, seed=seed, stop_criterion=max_change)
        plot_state_continuous(logfile, figname=f"BRIM_afoakwa_{bench}_iter{num_iterations}.png")
        logger.info("BRIM afoakwa done")

        plot_energies_multiple(current_logfiles, figName=f"{bench}_energy_iter{num_iterations}.png", save_folder=figtop, best_found=best_found)
    
    plot_energy_dist_multiple_solvers(logfiles, xlabel="num_iterations", fig_name=f"{bench}_energy_all.png", save_folder=figtop,
                                      best_found=np.array(best))


def plot_state_continuous(logfile:pathlib.Path, figname:str, save:bool=True):
    """Plots the continuous state of the current run of a solver.
    It only accepts the following continuous state solvers :
        - BRIM
        - Simulated Bifurcation (discrete and ballistic version)
    The states are then plotted as continuous functions of the iteration.

    Args:
        logfile (pathlib.Path): The logfile in which the data of the solver is stored
        figname (str): the name of the figure the plot should be saved as
        save (bool, optional): whether to save the figure or not. Defaults to True.
    """
    if return_metadata(logfile, "solver") == "bSB":
        states = return_data(logfile, "positions")
    else:
        states = return_data(logfile, "voltages")

    plt.figure()
    plt.plot(states)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('continuous state')
    if save:
        plt.savefig(figtop / figname)
    plt.close()

if __name__ == "__main__":
    # test_small()
    # test_afoakwa()
    # test1()
    # test2()
    test_afoakwa2()
