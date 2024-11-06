import numpy as np
import matplotlib.pyplot as plt

def get_random_s(N: int)-> np.ndarray:
    """
    Creates a random initial vector of spins
    
    :param int N: number of spins in problem
    :return s (np.ndarray): vector of spins
    """
    return np.random.choice([-1, 1], N)


def get_coeffs_from_array_MC(N: int, data: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
    From an array of data it creates the interaction and magnetic field weights J and h.

    :param int N: size of the problem
    :param np.ndarray data: data array assumed to have the following structure per row [node 1, node 2, edge weight]
    :return J (np.ndarray): matrix of the interaction weights
    :return h (np.ndarray): vector of the magnetic field weights
    """
    J = np.zeros((N, N))
    h = np.zeros((N,))
    for row in data:
        i, j, weight = int(row[0])-1, int(row[1])-1, row[2]
        J[i, j] = -weight
        J[j, i] = -weight
    return J/2, h

def compute_rx(init: float, end: float, S: int) -> float:
    """
    Computes the change rate needed for hyperparameter control in the SA and SCA algorithm.

    :param float init:  initial value of the hyperparameter
    :param float end: end value of the hyperparameter
    :param int S: total number of iterations
    :return rx (float): change rate of the hyperparameter
    """
    return (end/init)**(1/(S-1))

def plot_energies(energies:dict[str:np.ndarray], S:int, filename:str)->None:
    """
    Plots the given energies and stores them in directory filename.

    :param dict[str:np.ndarray] energies: all the energies of solvers stored in a dictionary with the structure [solver, energy]
    :param int S: total amount of iterations
    :param str filename: absolute directory of the path to the file
    """
    title = ' '
    plt.figure()
    for key in energies.keys():
        title += f'{key}, '
        plt.plot(list(range(S)), energies[key], label=key)
    title = title[:-2]
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.title('Energy evolution of' + title)
    plt.savefig(filename)

def plot_energy_dist(energy:np.ndarray, nb_runs:int, solver:str, filename:str)->None:
    """
    Plots the optimal energies over different runs as a histogram.

    :param np.ndarray energy: optimal energies over all the runs
    :param int nb_runs: number of runs
    :param str filename: absolute directory of the path to the file
    """
    plt.figure()
    plt.hist(energy, int(nb_runs/10))
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title(f'{solver} energy outline')
    plt.savefig(filename)