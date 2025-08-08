import matplotlib.pyplot as plt
import numpy as np
import pathlib

from ising.flow import TOP
from ising.postprocessing.helper_functions import get_metadata_from_logfiles, compute_averages_energies


def plot_energy_distribution(
    logfiles: list[pathlib.Path],
    benchmark: str,
    figName: str = "benchmark_distribution_sweep",
    save: bool = True,
    save_dir: pathlib.Path = ".",
    percentage: float = 1.0,
):
    optimal_values = np.loadtxt(TOP / f"ising/benchmarks/{benchmark}/optimal_energy.txt", dtype=str)
    optimal_values = optimal_values[:int(len(optimal_values) * percentage), :]
    benchmarks = optimal_values[:, 0]
    data = get_metadata_from_logfiles(logfiles, "model_name", "solution_energy")

    plt.figure(constrained_layout=True)
    for solver in data:
        relative_errors = []
        for i, benchmark in enumerate(benchmarks):
            energies = data[solver][benchmark]
            optimal_value = float(optimal_values[i, 1])

            relative_errors.append(np.abs(energies - optimal_value) / np.abs(optimal_value))
        plt.boxplot(relative_errors, positions=range(len(benchmarks)), label=solver)
    plt.xticks(range(len(benchmarks)), benchmarks, rotation=45)
    plt.yscale('log')
    plt.xlabel("Benchmark")
    plt.ylabel("Relative error with optimal value of benchmark")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_dir / f"{figName}.pdf")
    plt.close()


def plot_energy_average(
    logfiles: list[pathlib.Path],
    benchmark: str,
    figName: str = "benchmark_average_sweep",
    save: bool = True,
    save_dir: pathlib.Path = ".",
    percentage: float = 1.0,
):
    optimal_values = np.loadtxt(TOP / f"ising/benchmarks/{benchmark}/optimal_energy.txt", dtype=str)
    benchmarks = optimal_values[:int(len(optimal_values) * percentage), 0]
    data = get_metadata_from_logfiles(logfiles, "model_name", "solution_energy")
    avg_energies, min_energies, max_energies, x = compute_averages_energies(data)
    plt.figure(constrained_layout=True)
    for solver in avg_energies:
        for i, benchmark in enumerate(benchmarks):
            optimal_value = float(optimal_values[i, 1])
            avg_energies[solver][i] = np.abs(avg_energies[solver][i] - optimal_value) / np.abs(
                optimal_value
            )
            min_energies[solver][i] = np.abs(min_energies[solver][i] - optimal_value) / np.abs(
                optimal_value
            )
            max_energies[solver][i] = np.abs(max_energies[solver][i] - optimal_value) / np.abs(
                optimal_value
            )
        plt.semilogy(range(len(benchmarks)), avg_energies[solver], label=solver)
        plt.fill_between(
            benchmarks,
            min_energies[solver],
            max_energies[solver],
            alpha=0.2,
        )
    plt.xticks(range(len(benchmarks)), benchmarks, rotation=45)
    plt.xlabel("Benchmark")
    plt.ylabel("Relative error to optimal value of benchmark")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_dir / f"{figName}.pdf")
    plt.close()
