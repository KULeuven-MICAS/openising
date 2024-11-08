import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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


def compute_energy(J:np.ndarray, h:np.ndarray, sigma:np.ndarray)->float:
    """
    Computes the Hamiltonian given a sample.

    :param np.ndarray J: interaction coefficients
    :param np.ndarray h: magnetic field coefficients
    :param np.ndarray sigma: sample
    :return H (float): value of the Hamiltonian
    """
    return -np.inner(sigma.T, np.inner(J, sigma)) - np.inner(h.T, sigma)

def compute_energy_bSB(h, J, y, x, a0, at, c0):
    if all(map(np.abs(x[i]) <= 1 for i in range(np.shape(x)[0]))):
        V = (a0-at)/2*np.sum(np.power(x, 2)) - c0/2*np.inner(x.T, np.inner(J, x)) - c0*np.inner(h.T, x)
    else:
        V = np.inf
    return a0/2*np.sum(np.power(y, 2)) + V

def compute_energy_dSB(h, J, y, x, a0, at, c0):
    if all(map(np.abs(x[i]) <= 1 for i in range(np.shape(x)[0]))):
        V = (a0-at)/2*np.sum(np.power(x, 2)) - c0/2*np.inner(x.T, np.inner(J, np.sign(x))) - c0*np.inner(h.T, x)
    else:
        V = np.inf
    return a0/2*np.sum(np.power(y, 2)) + V

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


def plot_energy_dist(energy:np.ndarray, solver:str, filename:str)->None:
    """
    Plots the optimal energies over different runs as a histogram.

    :param np.ndarray energy: optimal energies over all the runs
    :param str filename: absolute directory of the path to the file
    """
    plt.figure()
    plt.hist(energy, 10)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title(f'{solver} energy outline')
    plt.savefig(filename)


def add_edges_graph(graph:nx.Graph, states:np.ndarray|list, G_orig:nx.Graph)->tuple[list, list, dict]:
    """
    Splits the edges of the graph into two parts according to the state value and adds the edges accordingly.

    :param nx.Graph graph: the new graph
    :param np.ndarray states: optimal state
    :param nx.Graph G_orig: original graph
    :return red_nodes (list): list of all the nodes with value +1
    :return blue_nodes (list): list of all the nodes with value -1 or 0
    :return labels (dict): dictionary with all the labels of all the nodes
    """
    red_nodes = []
    blue_nodes = []
    labels = {}
    for i in range(len(states)):
        graph.add_node(i)
        labels[i] = i
        if states[i] == 1:
            red_nodes.append(i)
        else:
            blue_nodes.append(i)
        for j in range(len(states)):
            if i != j and states[i] == states[j] and G_orig.has_edge(i, j):
                graph.add_edge(i, j)
    return red_nodes, blue_nodes, labels


def plot_solution(state:np.ndarray|list, G_orig:nx.Graph, solver:str)->None:
    """
    Plots the Graph solution of the problem

    :param np.ndarray state: the optimal state of the problem
    :param nx.Graph G_orig: original graph
    :param str solver: solver that was used to solve the problem
    """     
    G = nx.Graph()
    red_nodes, blue_nodes, labels = add_edges_graph(G, state, G_orig)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red')
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.title(solver)