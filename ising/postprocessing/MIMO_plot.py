import numpy as np
import matplotlib.pyplot as plt
import pathlib

from ising.generators.MIMO import compute_difference
from ising.postprocessing.helper_functions import compute_averages_energies
from ising.utils.HDF5Logger import return_metadata


def plot_error_SNR(
    logfiles: dict[int : dict[str : pathlib.Path]],
    M: int,
    actual_solutions: dict[int : dict[int:np.ndarray]],
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figname: str = "error_SNR.png",
):
    """Plots the relative error between the optimal solution and the computed solution for different SNRs.

    Args:
        logfiles (dict[int: dict[str:pathlib.Path]]): the logfiles for the different SNRs.
        M (int): the considered QAM scheme.
        actual_solutions (list[np.ndarray]): the actual solutions
    """

    error_data = dict()
    for SNR, solver in logfiles.items():
        error_data[SNR] = dict()
        for solver_name, logfiles in solver.items():
            optim_states = [return_metadata(logfile, metata="solution_state") for logfile in logfiles]
            error_data[SNR][solver_name] = []
            i=0
            for optim_state in optim_states:
                error_data[SNR][solver_name].append(compute_difference(optim_state, actual_solutions[SNR][i], M))
                i+=1
    avg_error, min_error, max_error, x_data = compute_averages_energies(error_data)

    plt.figure()
    for solver_name, error in avg_error.items():
        plt.plot(x_data, error, label=solver_name)
        plt.fill_between(x_data, min_error[solver_name], max_error[solver_name], alpha=0.2)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Bit Error Rate")
    plt.legend()
    if save:
        plt.savefig(save_folder / figname)
    plt.close()
