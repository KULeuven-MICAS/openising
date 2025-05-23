import matplotlib.pyplot as plt
import pathlib

from ising.postprocessing.helper_functions import compute_averages_energies, get_metadata_from_logfiles


def plot_error_SNR(
    logfiles: list[pathlib.Path],
    gurobi_files: list[pathlib.Path]|None =None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figName: str = "error_SNR",
) -> None:
    """Plots the relative error between the optimal solution and the computed solution for different SNRs.

    Args:
        logfiles (list[pathlib.Path]): List of all the logfiles for the different solvers.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): The path to the folder in which to save the figure. Defaults to ".".
        figName (str, optional): The name of the figure to save. Defaults to error_SNR.
    """
    if gurobi_files is not None:
        gurobi_data = get_metadata_from_logfiles(gurobi_files, x_data="SNR", y_data="BER")
        gurobi_avg, _, _, gurobi_x = compute_averages_energies(gurobi_data)
    data = get_metadata_from_logfiles(logfiles, x_data="SNR", y_data="BER")
    avg_error, _, _, x_data = compute_averages_energies(data)
    plt.figure()
    for solver_name, error in avg_error.items():
        plt.semilogy(x_data[solver_name], error, label=solver_name)
    if gurobi_files is not None:
        plt.semilogy(gurobi_x["Gurobi"],gurobi_avg["Gurobi"], linestyle="--", label="Gurobi")
    plt.xlabel("SNR [dB]")
    plt.xticks(x_data[solver_name])
    plt.ylabel("Bit Error Rate")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_folder / f"{figName}.pdf", dpi=600, bbox_inches="tight")
    plt.close()
