import matplotlib.pyplot as plt
import pathlib

from ising.postprocessing.helper_functions import return_data, return_metadata

def plot_data(logfile:pathlib.Path, data_name:str, figName:str, save:bool=True, save_folder:pathlib.Path='.'):
    """PLots a certain logged data over the iterations.

    Args:
        logfile (pathlib.Path): The absolute path to the log file containing the data.
        data_name (str): The name of the data to be plotted.
        fig_name (str): The name of the figure to be saved.
        save (bool, optional): whether to save the figure or not. Defaults to True.
        save_folder (pathlib.Path, optional): absolute path to the folder where the figure should be stored.
                                             Defaults to '.'.
    """
    data = return_data(logfile, data_name)
    num_iterations = return_metadata(logfile, "num_iterations")

    plt.figure()
    plt.plot(list(range(num_iterations)), data)
    plt.xlabel("Iterations")
    plt.ylabel(data_name)
    plt.title(f"{data_name} over iterations")
    if save:
        plt.savefig(save_folder / f"{figName}.pdf")
    plt.close()
