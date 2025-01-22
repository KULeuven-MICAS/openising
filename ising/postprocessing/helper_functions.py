import pathlib
import numpy as np

from ising.utils.HDF5Logger import return_metadata, return_data


def compute_averages_energies(
    data: dict[str : dict[float : np.ndarray]],
) -> tuple[dict[str : np.ndarray], dict[str : np.ndarray], dict[str : np.ndarray], dict[str:float]]:
    """This function computes the average and std of some data.
    More precisely, the averages and std's are computed for each solver and x-axis data point seperately.
    It is very important the given data is structured as follows:
        - The dictionary needs to be ordened per solver
        - Each solver has a dictionary as its value. This dictionary is ordened per x-axis data point.
        - The value for each x-axis data point is a list of all the possible y-axis data point (due to multiple runs).

    Args:
        data (dict[str : dict[float : np.ndarray]]): A dictionary conatining all the data,
                                                     which is structured as explained above.

    Returns:
        _type_: _description_
    """
    x_data = {solver : [x_dat for (x_dat, _) in data[solver].items()] for solver in data.keys()}
    avg_energies = {solver: [] for solver in data.keys()}
    min_energies = {solver: [] for solver in data.keys()}
    max_energies = {solver: [] for solver in data.keys()}

    for solver_name, plot_info in data.items():
        for _, y_data in plot_info.items():
            avg_energies[solver_name].append(np.mean((y_data), axis=0))
            std = np.std(y_data, axis=0)
            min_energies[solver_name].append(avg_energies[solver_name][-1] - std)
            max_energies[solver_name].append(avg_energies[solver_name][-1] + std)
            # x_data[solver_name].append(x_dat)

        # make sure the data is a numpy array
        avg_energies[solver_name] = np.array(avg_energies[solver_name])
        min_energies[solver_name] = np.array(min_energies[solver_name])
        max_energies[solver_name] = np.array(max_energies[solver_name])

    return avg_energies, min_energies, max_energies, x_data


def get_metadata_from_logfiles(
    logfiles: list[pathlib.Path], x_data: str, y_data: str
) -> dict[str : dict[float : np.ndarray]]:
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
        dict[str, dict[float : np.ndarray]]: the data dictionary with the correct y data for each solver and x value.
    """
    data = {
        solver: {x: [] for x in [(return_metadata(logfile, x_data)) for logfile in logfiles
                                                                    if return_metadata(logfile, "solver") == solver]
                    }
        for (solver) in [return_metadata(logfile, "solver") for logfile in logfiles]
    }
    for logfile in logfiles:
        solver = return_metadata(logfile, "solver")
        x = return_metadata(fileName=logfile, metadata=x_data)
        y = return_metadata(fileName=logfile, metadata=y_data)
        data[solver][x].append(y)
    data = {solver: {x:np.array(y) for (x, y) in data[solver].items()} for solver in data.keys() }

    return data


def get_data_from_logfiles(
    logfiles: list[pathlib.Path],
    y_data: str,
) -> dict[str : dict[float : np.ndarray]]:
    """Generates a dictionary with the correct y data for each solver.
    The dictionary is structured as follows:
        - The dictionary is ordened per solver
        - each solver has a dictionary as its value which only have one key-value pair
        - the key for each solver is 1, their value is a list of all the y-data.

    Args:
        logfiles (list[pathlib.Path]): A list of all the logfiles that need to be looked at
        y_data (str): the data that needs to be stored in the dictionary

    Returns:
        dict[str:dict[int:list[np.ndarray]]]: The dictionary consists of all the data
    """

    data = {solver: {1.: []} for solver in [return_metadata(logfile, "solver") for logfile in logfiles]}
    for logfile in logfiles:
        solver = return_metadata(logfile, "solver")
        y = return_data(fileName=logfile, data=y_data)

        data[solver][1.].append(y)
    data = {(solver, {x:np.array(y)}) for solver in data.keys() for (x, y) in data[solver].items()}
    return data
