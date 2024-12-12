import pathlib
import h5py
import numpy as np

def return_data(fileName: pathlib.Path, data: str) -> np.ndarray:
    with h5py.File(fileName, "r") as logfile:
        data = logfile[data][:]
    return data

def return_metadata(fileName:pathlib.Path, metadata:str):
    with h5py.File(fileName, "r") as logfile:
        metadata = logfile.attrs[metadata]
    return metadata

def compute_averages_energies(data:dict[int:dict[str:np.ndarray]]):
    x_data = []
    avg_energies = dict()
    min_energies = dict()
    max_energies = dict()

    for x_dat, solver_info in data.items():
        for solver_name, energy in solver_info.items():
            if solver_name not in avg_energies:
                avg_energies[solver_name] = []
                min_energies[solver_name] = []
                max_energies[solver_name] = []
            avg_energies[solver_name].append(np.mean(np.array(energy), axis=0))
            std = np.std(energy)
            min_energies[solver_name].append(avg_energies[solver_name][-1] - std)
            max_energies[solver_name].append(avg_energies[solver_name][-1] + std)
        x_data.append(x_dat)
    return avg_energies, min_energies, max_energies, x_data


def get_bestEnergy_from_dict(logfiles:dict[int, dict[str, list[pathlib.Path]]], y_data:str):
    data = dict()
    for xdata, solver in logfiles.items():
        data[xdata] = {}
        for solver_name, logfile_list in solver.items():
            for logfile in logfile_list:
                if solver_name not in data[xdata]:
                    data[xdata][solver_name] = []
                data[xdata][solver_name].append(return_metadata(fileName=logfile, metadata=y_data))
    return data
