import numpy as np
import pathlib

from ising.utils.HDF5Logger import return_metadata

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

