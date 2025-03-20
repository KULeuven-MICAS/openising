import matplotlib.pyplot as plt
import pathlib

from ising.utils.HDF5Logger import return_data, return_metadata

def plot_state(solver:str, logfile:pathlib.Path, figname:str, figtop:pathlib.Path="."):
    """Delegates the plotting of the state to the correct function.

    Args:
        solver (str): the solver of which the state should be plotted.
        logfile (pathlib.Path): the logfile in which the data is stored.
        figname (str): the name of the figure.
        figtop (pathlib.Path, optional): the absolute path to where the figure should be stored. Defaults to ".".
    """
    if solver in ["BRIM", "bSB", "dSB", "Multiplicative"]:
        plot_state_continuous(logfile=logfile, figname=figname, save_folder=figtop)
    else:
        plot_state_discrete(logfile=logfile, figname=figname, save_folder=figtop)


def plot_state_discrete(logfile:pathlib.Path, figname:str, save:bool=True, save_folder:pathlib.Path='.'):
    """Plots the discrete state of the current run of a solver.
    The state at each iteration is plotted as a heatmap.

    Args:
        logfile (pathlib.Path): absolute path to the logfile in which everything is logged.
        figname (str): the name of the figure that should be saved.
        save (bool, optional): whether to save the figure or not. Defaults to True.
        save_folder (pathlib.Path, optional): absolute path to the destination folder where the figure should be stored.
                                              Defaults to '.'.
    """
    sigma = return_data(logfile, 'state').T
    plt.figure()
    plt.imshow(sigma,cmap='hot', interpolation='none', aspect='auto')
    plt.xlabel("iteration")
    plt.ylabel("sample")
    if save:
        plt.savefig(save_folder / figname)
    plt.close()

def plot_state_continuous(logfile:pathlib.Path, figname:str, save:bool=True, save_folder:pathlib.Path='.'):
    """Plots the continuous state of the current run of a solver.
    It only accepts the following continuous state solvers :
        - BRIM
        - Simulated Bifurcation (discrete and ballistic version)
    The states are then plotted as continuous functions of the iteration.

    Args:
        logfile (pathlib.Path): The logfile in which the data of the solver is stored
        figname (str): the name of the figure the plot should be saved as
        save (bool, optional): whether to save the figure or not. Defaults to True.
        save_folder (pathlib.Path, optional): the absolute path to the folder where the figure should be stored.
                                              Defaults to '.'.
    """
    solver = return_metadata(logfile, 'solver')
    # num_iterations = return_metadata(logfile, 'num_iterations')
    state_name = {"BRIM": "voltages", "Multiplicative": "voltages", "dSB": "positions", "bSB": "positions"}
    states = return_data(logfile, state_name[solver] if solver in state_name else "voltages")

    plt.figure()
    plt.plot(states)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('continuous state')
    if save:
        plt.savefig(save_folder / figname)
    plt.close()
