import pathlib
import matplotlib.pyplot as plt
import numpy as np

from ising.postprocessing.helper_functions import (
    get_metadata_from_logfiles,
    get_data_from_logfiles,
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
    logfiles: list[pathlib.Path],
    figName: str = "multiple_energies.png",
    y_data:str="energy",
    best_found: float = None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    percentage: float = 1.0
):
    """Plots the energies of multiple optimisation processes.

    Args:
        logfiles (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "multiple_energies.png".
        y_data (str, optional): the data that is used for the y axis. Defaults to "energy".
        best_found (float, optional): best found energy value of the problem. Defaults to 0.0.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where the figure should be stored. Defaults to ".".
        percentage (float, optional): percentage of the last iterations that will be plotted. Defaults to 100%.
    """
    #TODO: voeg gurobi toe
    data = get_data_from_logfiles(logfiles, y_data=y_data, x_data="num_iterations")
    avg_energies, min_energies, max_energies, _ = compute_averages_energies(data)

    percentage_plot = 1.0 - percentage
    plt.figure()
    for solver in avg_energies.keys():
        num_iterations = len(avg_energies[solver][0])
        iterations = range(percentage_plot*num_iterations, num_iterations)
        plt.semilogx(iterations,
                    avg_energies[solver][0][percentage_plot*num_iterations:, :], label=solver)
        plt.fill_between(
            iterations,
            min_energies[solver][0][percentage_plot*num_iterations:, :],
            max_energies[solver][0][percentage_plot*num_iterations:, :],
            alpha=0.2
        )
    if best_found is not None:
        plt.axhline(best_found, linestyle="--", color="k", label="Best Found")
        plt.axhline(0.99*best_found, linestyle="-.", color="k", label="0.99 of Best Found")
        plt.axhline(0.9*best_found, linestyle="-.", color="k", label="0.9 of Best Found")

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
    y_data:str = "solution_energy",
    fig_name: str = "multiple_solvers_energy_dist.png",
    best_found: np.ndarray | None = None,
    best_Gurobi: np.ndarray=None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """Plots the best found energy distribution from multiple runs and iteration lengths for multiple solvers.

    Args:
        logfiles (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "multiple_solvers_energy_dist.png".
        best_found (np.ndarray,None, optional): numpy ndarray of the best found solutions of the problem.
                                                Defaults to None.
        best_Gurobi (np.ndarray, optional): Numpy ndarray with the best solution of Gurobi. Defaults to None.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
    """
    data = get_metadata_from_logfiles(logfiles=logfiles, x_data=xlabel, y_data=y_data)
    avg_energies, min_energies, max_energies, x_data = compute_averages_energies(data)

    plt.figure(constrained_layout=True)
    for solver_name, _ in avg_energies.items():
        plt.semilogx(x_data[solver_name], avg_energies[solver_name], label=f"{solver_name}")
        plt.fill_between(x_data[solver_name], min_energies[solver_name], max_energies[solver_name], alpha=0.2)
    if best_found is not None:
        plt.semilogx(
            x_data[solver_name], best_found, "--k", label="Best found")
        plt.semilogx(x_data[solver_name], 0.99*best_found, "-.", color="k", label="0.99 of best found")
    if best_Gurobi is not None:
        plt.semilogx(x_data[solver_name], best_Gurobi, "--r", label="Best Gurobi")
    plt.xlabel(xlabel.replace("_", " "))
    plt.ylabel(y_data.replace("_", " "))
    plt.legend()
    if save:
        plt.savefig(save_folder / fig_name)
    plt.close()


def plot_relative_error(
    logfiles: list[pathlib.Path],
    best_found: np.ndarray,
    x_label: str,
    y_data:str="solution_energy",
    fig_name: str = "relative_error.png",
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """Plots the relative error of different solvers with the best found solution.
    It takes into account a solver can be run multiple times with the same parameters set.
    This results in a plot that could show a distribution of results per x point.

    Args:
        logfiles (list[pathlib.Path]): all the logfiles
        best_found (np.ndarray): an numpy ndarray holding the best found solution of the problem.
        x_data (str): the metadata that is used for the x axis.
        y_data (str, optional): the metadata that is used for the y axis. Defaults to "solution_energy".
        fig_name (str, optional): name of the figure to be saved. Defaults to "relative_error.png".
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
    """
    #TODO: gurobi toevoegen
    data = get_metadata_from_logfiles(logfiles, y_data=y_data, x_data=x_label)
    avg_energies, min_energies, max_energies, x_data = compute_averages_energies(data)

    plt.figure(constrained_layout=True)
    best_found = np.array(best_found)
    for solver_name in data.keys():
        relative_error = np.abs((avg_energies[solver_name] - best_found) / best_found)
        min_rel_error = np.abs((min_energies[solver_name] - best_found) / best_found)
        max_rel_error = np.abs((max_energies[solver_name] - best_found) / best_found)

        plt.loglog(x_data[solver_name], relative_error, label=f"{solver_name}")
        plt.fill_between(x_data[solver_name], min_rel_error, max_rel_error, alpha=0.2)
    plt.xlabel(x_label.replace("_", " "))
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
    """Plots the energy change over time of a solver.

    Args:
        logfile (pathlib.Path): the logfile of the solver.
        best_found (float | None, optional): the best found value of the problem. Defaults to None.
        save (bool, optional): Whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
        figName (str, optional): the name of the figure to save. Defaults to "energy_time.png".
    """
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
