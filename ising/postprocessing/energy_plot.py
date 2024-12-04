import pathlib
import matplotlib.pyplot as plt
import numpy as np

from ising.postprocessing.helper_functions import return_data, return_metadata

def plot_energies_on_figure(energies: np.ndarray, label: str | None = None):
    if label == "Best found":
        shape = "--"
    plt.plot(list(range(energies.shape[0])), energies, shape, label=label)


def plot_energies(fileName: pathlib.Path, best_found:float=0., save: bool = True, save_folder: pathlib.Path = "."):
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
        plt.savefig(f"{save_folder}/{solver_name}_energy.png")

    plt.show()


def plot_energies_multiple(fileName_list: list[pathlib.Path], best_found:float=0., save: bool = True,
                           save_folder: pathlib.Path = "."):
    plt.figure()
    title = ""
    for fileName in fileName_list:
        energies, best_energy, solver_name = (
            return_data(fileName=fileName, data="energy"),
            return_metadata(fileName, metadata="solution_energy"),
            return_metadata(fileName, metadata="solver"),
        )

        plot_energies_on_figure(energies, label=f"{solver_name} (Best: {best_energy})")
        title += solver_name + ", "
    if best_found != 0.:
        plot_energies_on_figure(np.ones(energies.shape) * best_found, label="Best found")
    plt.legend()
    plt.title(f"Energies of {title[:-2]}")
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    if save:
        plt.savefig(save_folder / "multiple_energies.png")
    plt.show()


def plot_energy_dist(fileName_list: list[pathlib.Path], best_found:float=0.,
                     save: bool = True, save_folder: pathlib.Path = "."):
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
    plt.fill_between(min_best_energies, max_best_energies, color ='blue', alpha=0.2)
    if best_found != 0.:
        plt.plot(num_iters, np.ones(len(num_iters)) * best_found, label="Best found")
    plt.title(f"Average Best Energy of {solver} with Standard Deviation")
    plt.xlabel("iteration")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / f"{solver}_best_energy_distribution.png")
    plt.show()


def plot_energy_dist_multiple_solvers(
    fileName_list: list[pathlib.Path], best_found:float=0., save: bool = True, save_folder: pathlib.Path = "."
):
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
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
        plt.fill_between(
            num_iters, min_best_energies, max_best_energies, color=color, alpha=0.2
        )
    if best_found != 0.:
        plt.plot(num_iters, np.ones(len(num_iters)) * best_found, label="Best found")
    plt.title("Average Best Energy with Standard Deviation for Multiple Solvers")
    plt.xlabel("iteration")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(save_folder / "best_energy_distribution_multiple_solvers.png")
    plt.show()
