import matplotlib.pyplot as plt
import pathlib

from ising.postprocessing.helper_functions import return_data, return_metadata

def plot_state_discrete(logfile:pathlib.Path, figName:str, save:bool=True, save_folder:pathlib.Path='.'):
    sigma = return_data(logfile, 'state').T

    plt.figure()
    plt.imshow(sigma, cmap='hot', interpolation='nearest')
    if save:
        plt.savefig(save_folder / figName)

def plot_state_continuous(logfile:pathlib.Path, figname:str, save:bool=True, save_folder:pathlib.Path='.'):
    solver = return_metadata(logfile, 'solver')
    num_iterations = return_metadata(logfile, 'num_iterations')
    if solver == "BLIM":
        states = return_data(logfile, 'voltages')
    else:
        states = return_data(logfile, 'positions')

    plt.figure()
    plt.plot(list(range(num_iterations)), states)
    plt.xlabel('Iteration')
    plt.ylabel('continuous state')
    if save:
        plt.savefig(save_folder / figname)
