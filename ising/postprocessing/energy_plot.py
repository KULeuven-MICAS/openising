import pathlib
import matplotlib.pyplot as plt
import numpy as np

from ising.postprocessing.helper_functions import (
    get_metadata_from_logfiles,
    compute_averages_energies,
)

from ising.utils.HDF5Logger import return_data, return_metadata


def plot_energies_on_figure(energies: np.ndarray, label: str | None = None):
    """PLots the energies on a given figure.

    Args:
        energies (np.ndarray): the energies that need to be plotted.
        label (str | None, optional): label of the plot. Defaults to None.
    """
    if label == "Best found":
        shape = "--"
    else:
        shape = "-"
    plt.plot(energies, shape, label=label)


def plot_energies(
    fileName: pathlib.Path,
    figName: str = "energies.png",
    best_found: float = 0.0,
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """Plots the energies of a given optimisation process using the logfile.

    Args:
        fileName (pathlib.Path): absolute path to the logfile.
        figName (str, optional): name of the figure that should be saved. Defaults to "energies.png".
        best_found (float, optional): Best found energy value of the problem. Defaults to 0.0.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): Folder to which the figure should be saved. Defaults to ".".
    """
    energies, best_energy, solver_name = (
        return_data(fileName=fileName, data="energy"),
        return_metadata(fileName, metadata="solution_energy"),
        return_metadata(fileName, metadata="solver"),
    )

    plt.figure()
    plot_energies_on_figure(energies, label=solver_name)
    plot_energies_on_figure(np.ones(energies.shape) * best_found, label="Best found")
    plt.title(f"Best energy: {best_energy}")
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    if save:
        plt.savefig(save_folder / figName)
    plt.close()


def plot_energies_multiple(
    fileName_list: list[pathlib.Path],
    figName: str = "multiple_energies.png",
    best_found: float = 0.0,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    diff_metadata: str | None = None,
):
    """Plots the energies of multiple optimisation processes.

    Args:
        fileName_list (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "multiple_energies.png".
        best_found (float, optional): best found energy value of the problem. Defaults to 0.0.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where the figure should be stored. Defaults to ".".
    """
    plt.figure()
    for fileName in fileName_list:
        energies, best_energy, solver_name = (
            return_data(fileName=fileName, data="energy"),
            return_metadata(fileName, metadata="solution_energy"),
            return_metadata(fileName, metadata="solver"),
        )
        if diff_metadata is not None:
            diff = return_metadata(fileName, metadata=diff_metadata)
            solver_name += f" {diff_metadata}: {diff}"
        plot_energies_on_figure(energies, label=f"{solver_name} (Best: {best_energy})")
    if best_found != 0.0:
        plot_energies_on_figure(np.ones(energies.shape) * best_found, label="Best found")
    plt.legend()
    plt.title("Energy comparison of different optimisation processes")
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    if save:
        plt.savefig(save_folder / figName)
    plt.close()


def plot_energy_dist_multiple_solvers(
    logfiles: list[pathlib.Path],
    xlabel: str,
    fig_name: str = "multiple_solvers_energy_dist.png",
    best_found: list[float] | None = None,
    best_Gurobi: bool = False,
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """Plots the best found energy distribution from multiple runs and iteration lengths for multiple solvers.

    Args:
        logfiles (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "multiple_solvers_energy_dist.png".
        best_found (list[float],None, optional): list of the best found solutions of the problem. Defaults to None.
        best_Gurobi (bool, optional): whether the best found solution is from Gurobi solver. Defaults to False.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
    """
    data = get_metadata_from_logfiles(logfiles=logfiles, x_data=xlabel, y_data="solution_energy")
    avg_energies, min_energies, max_energies, x_data = compute_averages_energies(data)

    plt.figure()
    for solver_name, _ in avg_energies.items():
        plt.semilogx(x_data[solver_name], avg_energies[solver_name], label=f"{solver_name}")
        plt.fill_between(x_data[solver_name], min_energies[solver_name], max_energies[solver_name], alpha=0.2)
    if best_found is not None:
        plt.semilogx(
            x_data[solver_name], best_found, "--k", label="Best found: Gurobi" if best_Gurobi else "Best found"
        )
    plt.xlabel(xlabel)
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / fig_name)
    plt.close()


def plot_relative_error(
    logfiles: list[pathlib.Path],
    best_found: list[float],
    x_data: str,
    fig_name: str = "relative_error.png",
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    data = get_metadata_from_logfiles(logfiles, x_data, y_data="solution_energy")
    avg_energies, min_energies, max_energies, x_data = compute_averages_energies(data)

    plt.figure()
    for solver_name, _ in avg_energies.items():
        relative_error = (avg_energies[solver_name] - best_found) / best_found
        min_rel_error = (min_energies[solver_name] - best_found) / best_found
        max_rel_error = (max_energies[solver_name] - best_found) / best_found
        plt.loglog(x_data[solver_name], relative_error, label=f"{solver_name}")
        plt.fill_between(x_data[solver_name], min_rel_error, max_rel_error, alpha=0.2)
    plt.xlabel(x_data)
    plt.ylabel("Relative error with best found")
    plt.legend()
    if save:
        plt.savefig(save_folder / fig_name)
    plt.close()


def plot_energy_time(
    logfile: pathlib.Path,
    best_found: float | None = None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figName: str = "energy_time.png",
):
    time = return_data(logfile, "time_clock")
    energy = return_data(logfile, "energy")

    plt.figure()
    plt.plot(time, energy)
    plt.axhline(best_found, "--k", label="Best found")
    plt.title("Energy evolution over time")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / figName)
    plt.close()


def plot_energy_time_multiple(
    logfiles: list[pathlib.Path],
    best_found: float | None = None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figName: str = "energy_time.png",
):
    """Plots the average energy of multiple solvers over the run time.

    Args:
        logfiles (list[pathlib.Path]): list of the all the logfiles.
        best_found (float | None, optional): best found solution of the problem. Defaults to None.
        save (bool, optional): _description_. Defaults to True.
        save_folder (pathlib.Path, optional): _description_. Defaults to '.'.
        figName (str, optional): _description_. Defaults to "energy_time.png".
    """
    data = get_metadata_from_logfiles(logfiles, x_data="total_time", y_data="solution_energy")
    avg_energies, min_energies, max_energies, x_data = compute_averages_energies(data)

    plt.figure()
    for solver_name, energies in avg_energies.items():
        plt.semilogx(x_data[solver_name], energies, label=f"{solver_name}")
        plt.fill_between(x_data[solver_name], min_energies[solver_name], max_energies[solver_name], alpha=0.2)
    if best_found is not None:
        plt.axhline(
            y=best_found,
            color="k",
            linestyle="--",
            label="Best found",
        )
    plt.xlabel("time [s]")
    plt.ylabel("Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / figName)
    plt.close()
