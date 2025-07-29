import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from ising.utils.HDF5Logger import return_metadata
from ising.postprocessing.helper_functions import get_metadata_from_logfiles

def summary_energies(logfiles:list[pathlib.Path], save_dir:pathlib.Path) -> None:
    """Summarizes the energies over multiple sweeps for each solver and benchmark solved.
    The summary will hold the minimum, maximum, average and std values over the sweep.

    Args:
        logfiles (list[pathlib.Path]): a list of all the log files to summarize.
        save_dir (pathlib.Path): where to store the data.
    """
    energies = dict()

    for logfile in logfiles:
        solver_name = return_metadata(logfile, "solver")
        model_name = return_metadata(logfile, "model_name")

        if energies.get((solver_name, model_name)) is None:
            energies[(solver_name, model_name)] = []

        energy = return_metadata(logfile, "solution_energy")
        energies[((solver_name, model_name))].append(energy)

    header = "min max avg std"
    for (solver_name, model_name), all_energies in energies.items():
        summary = np.array([[np.min(all_energies),np.max(all_energies),np.mean(all_energies),np.std(all_energies)]])
        save_path = save_dir / f"{solver_name}_{model_name}_summary.csv"
        np.savetxt(save_path, summary, fmt="%.2f", header=header)

def box_plot_energies(logfiles:list[pathlib.Path], best_found: float, save_dir:pathlib.Path):
    data = get_metadata_from_logfiles(logfiles, "num_iterations", "solution_energy")

    df = {}
    for solver_name, info in data.items():
        for x_dat, y_dat in info.items():
            df[solver_name] = pd.DataFrame({"solver": solver_name, "energy":y_dat})
    df = pd.concat(list(df.values()))

    plt.figure()
    sns.boxplot(data=df, x="solver", y="energy")
    plt.axhline(y=best_found, color='k', linestyle="--", label=f'Best found: {best_found}')
    plt.legend()
    plt.savefig(save_dir / "boxplot_energies.png", bbox_inches="tight")
    plt.close()

