import pathlib
import numpy as np

from ising.utils.HDF5Logger import return_metadata


def compute_averages_energies(data: dict[str : dict[float : list[float]]]):
    x_data = dict()
    avg_energies = dict()
    min_energies = dict()
    max_energies = dict()

    for solver_name, plot_info in data.items():
        for x_dat, y_data in plot_info.items():
            if solver_name not in avg_energies:
                avg_energies[solver_name] = []
                min_energies[solver_name] = []
                max_energies[solver_name] = []
                x_data[solver_name] = []
            avg_energies[solver_name].append(np.mean(np.array(y_data), axis=0))
            std = np.std(y_data)
            min_energies[solver_name].append(avg_energies[solver_name][-1] - std)
            max_energies[solver_name].append(avg_energies[solver_name][-1] + std)
            x_data[solver_name].append(x_dat)

    return avg_energies, min_energies, max_energies, x_data


def get_metadata_from_logfiles(
    logfiles: list[pathlib.Path], x_data: str, y_data: str
) -> dict[str : dict[float : list[float]]]:
    """Generates a dictionary with the correct y data for each solver and x value.
    The dictionary is structured as follows:
        - the dictionary is ordened per solver
        - each solver has a dictionary with the x values as keys
        - for each x value the corresponding y values are stored in a list.
            There are multiple y values due to the multiple runs the solvers have done.

    Args:
        logfiles (list[pathlib.Path]): list of all the absolute paths to the logfiles.
        x_data (str): name of the x axis metadata that is needed for the plot
        y_data (str): name of the y axis metadata that is needed for the plot.

    Returns:
        dict[str, dict[float : list[float]]]: the data dictionary with the correct y data for each solver and x value.
    """
    data = dict()
    for logfile in logfiles:
        solver = return_metadata(fileName=logfile, metadata="solver")
        # Make sure the solver already exists in the data dictionary
        if solver not in data:
            data[solver] = dict()

        x = return_metadata(fileName=logfile, metadata=x_data)
        y = return_metadata(fileName=logfile, metadata=y_data)

        # Make sure the x value already exists in the data[solver] dictionary
        if x not in data[solver]:
            data[solver][x] = []
        data[solver][x].append(y)

    return data
