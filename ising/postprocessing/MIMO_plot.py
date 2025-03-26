import matplotlib.pyplot as plt
import pathlib

from ising.postprocessing.helper_functions import compute_averages_energies, get_metadata_from_logfiles


def plot_error_SNR(
    logfiles: list[pathlib.Path],
    gurobi_files: list[pathlib.Path]|None =None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figname: str = "error_SNR.png",
) -> None:
    """Plots the relative error between the optimal solution and the computed solution for different SNRs.

    Args:
        logfiles (list[pathlib.Path]): List of all the logfiles for the different solvers.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): The path to the folder in which to save the figure. Defaults to ".".
        figname (str, optional): The name of the figure to save. Defaults to error_SNR.png.
    """

    # error_data = dict()
    # for solver, solver_logfiles in logfiles.items():
    #     error_data[solver] = dict()
    #     for logfile in solver_logfiles:
    #         logfile_str = str(logfile)
    #         run = int(logfile_str.split("_")[-1].split('.')[0][3:])
    #         SNR = int(logfile_str.split("_")[-2][3:])
    #         solver = return_metadata(logfile, "solver")
    #         if SNR not in error_data[solver].keys():
    #             error_data[solver][SNR] = []
    #         xtilde = actual_solutions[SNR][run]
    #         optim_state = return_metadata(logfile, "solution_state")
    #         error_data[solver][SNR].append(compute_difference(optim_state, xtilde, M))

    if gurobi_files is not None:
        gurobi_data = get_metadata_from_logfiles(gurobi_files, x_data="SNR", y_data="BER")
        gurobi_avg, gurobi_min, gurobi_max, gurobi_x = compute_averages_energies(gurobi_data)
    data = get_metadata_from_logfiles(logfiles, x_data="SNR", y_data="BER")
    avg_error, min_error, max_error, x_data = compute_averages_energies(data)

    plt.figure()
    for solver_name, error in avg_error.items():
        plt.semilogx(x_data[solver_name], error, label=solver_name)
        # plt.fill_between(x_data[solver_name], min_error[solver_name], max_error[solver_name], alpha=0.2)
    if gurobi_files is not None:
        plt.semilogx(gurobi_x["Gurobi"],gurobi_avg["Gurobi"], linestyle="--", label="Gurobi")
        # plt.fill_between(gurobi_x["Gurobi"], gurobi_min["Gurobi"], gurobi_max["Gurobi"], alpha=0.2)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Bit Error Rate")
    plt.legend()
    if save:
        plt.savefig(save_folder / figname)
    plt.close()
