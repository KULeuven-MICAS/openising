import pathlib
import matplotlib.pyplot as plt
import numpy as np

from ising.postprocessing.helper_functions import return_data, return_metadata


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
    plt.show()


def plot_energies_multiple(
    fileName_list: list[pathlib.Path],
    figName: str = "multiple_energies.png",
    best_found: float = 0.0,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    diff_metadata:str|None = None
):
    """PLots the energies of multiple optimisation processes.

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
    plt.show()


def plot_energy_dist(
    fileName_list: list[pathlib.Path],
    figName: str = "energy_dist.png",
    best_found: float = 0.0,
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """Plots the best found energy distribution over multiple runs for multiple iteration lengths.

    Args:
        fileName_list (list[pathlib.Path]): list of al the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "energy_dist.png".
        best_found (float, optional): best found energy value of the problem. Defaults to 0.0.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): the folder in which the figure should be saved. Defaults to ".".
    """
    data = dict()
    solver = ""
    for fileName in fileName_list:
        best_energy = return_metadata(fileName=fileName, metadata="solution_energy")
        num_iter = return_metadata(fileName=fileName, metadata="num_iterations")
        solvername = return_metadata(fileName=fileName, metadata="solver")
        if solver == "":
            solver = solvername
        if solver == solvername:
            if num_iter not in data:
                data[num_iter] = [best_energy]
            else:
                data[num_iter].append(best_energy)
        else:
            print("Only one solver is allowed")
    avg_best_energies = []
    min_best_energies = []
    max_best_energies = []
    num_iters = []
    for num_iter, best_energies in data.items():
        all_best_energies = np.array(best_energies)
        avg_best_energies.append(np.mean(all_best_energies, axis=0))
        std = np.std(all_best_energies, axis=0)
        min_best_energies.append(avg_best_energies[-1] - std)
        max_best_energies.append(avg_best_energies[-1] + std)
        num_iters.append(num_iter)

    plt.figure()
    plt.plot(num_iters, avg_best_energies, color="--b", label=f"{solver} Average Best Energy")
    plt.fill_between(min_best_energies, max_best_energies, color="blue", alpha=0.2)
    if best_found != 0.0:
        plt.plot(num_iters, np.ones(len(num_iters)) * best_found, label="Best found")
    plt.title(f"Average Best Energy of {solver} with Standard Deviation")
    plt.xlabel("total number of iterations")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / figName)
    plt.show()


def plot_energy_dist_multiple_solvers(
    fileName_list: list[pathlib.Path],
    figName: str = "multiple_solvers_energy_dist.png",
    best_found: float = 0.0,
    save: bool = True,
    save_folder: pathlib.Path = ".",
):
    """PLots the best found energy distribution from multiple runs and iteration lengths for multiple solvers.

    Args:
        fileName_list (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        figName (str, optional): name of the figure that will be saved. Defaults to "multiple_solvers_energy_dist.png".
        best_found (float, optional): best found solution of the problem. Defaults to 0.0.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
    """
    data = dict()
    for fileName in fileName_list:
        best_energy = return_metadata(fileName=fileName, metadata="solution_energy")
        num_iter = return_metadata(fileName=fileName, metadata="num_iterations")
        solvername = return_metadata(fileName=fileName, metadata="solver")
        if solvername not in data:
            data[solvername] = {}
        if num_iter not in data[solvername]:
            data[solvername][num_iter] = [best_energy]
        else:
            data[solvername][num_iter].append(best_energy)

    plt.figure()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    for solver, iter_data in data.items():
        avg_best_energies = []
        min_best_energies = []
        max_best_energies = []
        num_iters = []
        for num_iter, best_energies in iter_data.items():
            all_best_energies = np.array(best_energies)
            avg_best_energies.append(np.mean(all_best_energies, axis=0))
            std = np.std(all_best_energies, axis=0)
            min_best_energies.append(avg_best_energies[-1] - std)
            max_best_energies.append(avg_best_energies[-1] + std)
            num_iters.append(num_iter)

        color = color_cycle[color_index % len(color_cycle)]
        color_index += 1

        plt.plot(num_iters, avg_best_energies, label=f"{solver} Average Best Energy", color=color)
        plt.fill_between(num_iters, min_best_energies, max_best_energies, color=color, alpha=0.2)
    if best_found != 0.0:
        plt.plot(num_iters, np.ones(len(num_iters)) * best_found, label="Best found")
    plt.title("Average Best Energy with Standard Deviation for Multiple Solvers")
    plt.xlabel("total number of iterations")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / figName)
    plt.show()


def plot_energy_accuracy_check(
    logfiles: dict[int:pathlib.Path],
    best_found: list[float] | None = None,
    save: bool = True,
    save_folder: pathlib.Path = ".",
    figName: str = "energy_accuracy_check.png",
):
    """Plots the best found energy distribution compared to the best found from OpenJij.
    This plot does not care about iteration length but about proble size.

    Args:
        logfiles (list[pathlib.Path]): Dictionary of all the absolute paths to the logfiles linked to the problem size.
        best_found (np.ndarray | None, optional): List of the best found energy value according to OpenJij.
                                                  Defaults to None.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to ".".
        figName (str, optional): name of the figure that needs to be saved. Defaults to "energy_Accuracy_check.png".
    """
    energies = dict()

    for N, logfile_list in logfiles.items():
        energies[N] = []
        for logfile in logfile_list:
            energies[N].append(return_metadata(fileName=logfile, metadata="solution_energy"))

    average_energies = []
    min_energies = []
    max_energies = []
    problem_sizes = []

    for N, energy in energies.items():
        all_energies = np.array(energy)
        average_energies.append(np.mean(all_energies, axis=0))
        std = np.std(all_energies, axis=0)
        min_energies.append(average_energies[-1] - std)
        max_energies.append(average_energies[-1] + std)
        problem_sizes.append(N)

    plt.figure()
    plt.plot(problem_sizes, average_energies, '--b', label="Average Best Energy")
    plt.fill_between(problem_sizes, min_energies, max_energies, alpha=0.2)
    if best_found is not None:
        plt.plot(problem_sizes, best_found, 'k.-', label="Best found")
    plt.xlabel('problem size')
    plt.ylabel('Best Energy')
    plt.legend()
    plt.title("Average Best Energy with Standard Deviation over Different Problem Sizes")
    if save:
        plt.savefig(save_folder / figName)
    plt.show()

