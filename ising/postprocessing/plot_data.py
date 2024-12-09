import matplotlib.pyplot as plt
import pathlib

from ising.postprocessing.helper_functions import return_data, return_metadata

def plot_data(logFile:pathlib.Path, dataName:str, figName:str, save:bool=True, saveFolder:pathlib.Path='.'):
    data = return_data(logFile, dataName)
    num_iterations = return_metadata(logFile, "num_iterations")

    plt.figure()
    plt.plot(list(range(num_iterations)), data)
    plt.xlabel("Iterations")
    plt.ylabel(dataName)
    plt.title(f"{dataName} over iterations")
    if save:
        plt.savefig(saveFolder / figName)
    plt.show()
