import pathlib
import matplotlib.pyplot as plt
import h5py

def plot_energies(fileName:pathlib.Path, save:bool=True):
    logfile = h5py.File(fileName, 'r')
    energies = logfile.attrs['energies']
    best_energy = logfile.attrs['solution_energy']
    plt.figure()
    plt.plot(energies)
    plt.title(f"Best energy: {best_energy}")
    if save:
        plt.savefig(f"{logfile.attrs["solver"]}_energy.png")
