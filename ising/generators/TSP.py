import numpy as np
import networkx as nx
import time as t

from ising.stages.model.ising import IsingModel

__all__ = ["TSP"]


def TSP(graph: nx.DiGraph, weight_constant: float = 1.0) -> IsingModel:
    """Generates an Ising model of the assymetric TSP from the given directed graph

    Args:
        graph (nx.DiGraph): graph on which the TSP problem will be solved
        weight constant (float, optional): weight constant of the original objective function. Defaults to 1.

    Returns:
        model (IsingModel): Ising model of the TSP
    """
    if weight_constant < 1.0:
        raise ValueError("weight_constant should be larger than 1.0")
    N = len(graph.nodes)
    W = nx.linalg.adjacency_matrix(graph).toarray()
    maxW = np.max(W)
    constraint_weight = weight_constant * maxW

    # Add pseudo weights for non-existing connections
    for i in range(N):
        for j in range(N):
            if i != j and W[i, j] == 0:
                W[i,j] = weight_constant*maxW

    # Make a QUBO representation of the TSP problem
    J = np.zeros((N**2, N**2))
    h = np.zeros(N**2)
    constant = 0

    # Construct J
    # tour
    for u in range(N):
        for v in range(N):
            weight = W[u, v]
            for j in range(N):
                pos1 = get_index(j, u, N)
                pos2 = get_index(j+1, v, N)
                J[pos1, pos2] += -weight/2

    # time constraint
    for i in range(N):
        for u in range(N):
            pos1 = get_index(i, u, N)
            for v in range(N):
                pos2 = get_index(i, v, N)
                J[pos1, pos2] -= constraint_weight/2
    # place constraint
    for u in range(N):
        for i in range(N):
            pos1 = get_index(i, u, N)
            for j in range(N):
                pos2 = get_index(j, u, N)
                J[pos1, pos2] -= constraint_weight/2

    # construct h
    # tour
    for u in range(N):
        weight_sum = 0
        for v in range(N):
            weight_sum += W[u, v]
        for j in range(N):
            pos = get_index(j, u, N)
            h[pos] -= weight_sum/2
    # time constraint
    for i in range(N):
        for u in range(N):
            pos = get_index(i, u, N)
            h[pos] -= (N-2)/2*constraint_weight
    # place constraint
    for u in range(N):
        for i in range(N):
            pos = get_index(i, u, N)
            h[pos] -= (N-2)/2*constraint_weight

    for u in range(N):
        for v in range(N):
            weight = W[u, v]
            constant += weight*N/4
    constant += 2*N*((N/2 - 1)**2)*constraint_weight
    constant -= np.sum(np.diag(J))/2
    J = (J + J.T)/2
    J = np.triu(J, 1)
    return IsingModel(J, h, constant, name=graph.name)


def generate_random_TSP(
    N: int, seed: int = 0, weight_constant: float = 1.0, bit_width: int = 16
) -> tuple[IsingModel, nx.DiGraph]:
    if seed == 0:
        seed = int(t.time())

    np.random.seed(seed)
    coords_x = np.random.randint(0, int(2**bit_width - 1), N)
    coords_y = np.random.randint(0, int(2**bit_width - 1), N)

    graph = nx.DiGraph()
    node = 1
    for coordx, coordy in zip(coords_x, coords_y):
        graph.add_node(node, pos=(coordx, coordy))
        node += 1

    for i in range(N):
        for j in range(N):
            if i!= j:
                weight = compute_distance((coords_x[i], coords_y[i]), (coords_x[j], coords_y[j]))
                graph.add_edge(i+1, j+1, weight=weight)
    model = TSP(graph, weight_constant=weight_constant)
    return model, graph

def compute_distance(coords1: tuple[float, float], coords2: tuple[float, float])-> float:
    """Computes the Euclidean distance between two coordinates.

    Args:
        coords1 (tuple[float, float]): the first coordinate (x, y).
        coords2 (tuple[float, float]): the second coordinate (x, y).
    Returns:
        float: the Euclidean distance between the two coordinates.
    """
    return np.sqrt((coords1[0]-coords2[0])**2 + (coords1[1] - coords2[1])**2)

def get_index(time: int, city: int, N: int) -> int:
    """Returns the index of the ising spin corresponding to the city and time.
    The problem has N cities and time steps, making for N^2 spins. This corresponds to the following indexing rule:
        index = city * N + time
    This means the first N spins correspond to the first N time-steps of city 1, and so on.

    Args:
        time (int): the time step.
        city (int): the city.
        N (int): the amount of cities.

    Returns:
        int: the corresponding ising spin index
    """
    if time >= N:
        time -= N
    return (city * N) + time


def add_HA(J: np.ndarray, h: np.ndarray, W: np.ndarray, N: int):
    """Generates the objective function term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        W (np.ndarray): the weight matrix
        N (int): the amount of cities.
    """
    for u in range(N):
        for v in range(N):
            weight = W[u, v]
            for j in range(N):
                pos1 = get_index(j, u, N)
                pos2 = get_index(j+1, v, N)
                J[pos1, pos2] -= weight/2

    for u in range(N):
        weight_sum = 0
        for v in range(N):
            weight_sum += W[u, v]
        for j in range(N):
            pos = get_index(j, u, N)
            h[pos] -= weight_sum/2


def add_HB(J: np.ndarray, h: np.ndarray, N: int, B: float):
    """Generates the time constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities
        B (float): the time constraint constant.
    """
    for i in range(N):
        for u in range(N):
            pos1 = get_index(i, u, N)
            for v in range(N):
                pos2 = get_index(i, v, N)
                J[pos1, pos2] -= B/2
    for i in range(N):
        for u in range(N):
            pos = get_index(i, u, N)
            h[pos] -= (N-2)/2*B


def add_HC(J: np.ndarray, h: np.ndarray, N: int, C: float):
    """Generates the place constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities.
        C (float): the place constraint constant.
    """
    for u in range(N):
        for i in range(N):
            pos1 = get_index(i, u, N)
            for j in range(N):
                pos2 = get_index(j, u, N)
                J[pos1, pos2] -= C/2
    for u in range(N):
        for i in range(N):
            pos = get_index(i, u, N)
            h[pos] -= (N-2)/2*C


def get_TSP_value(graph: nx.DiGraph, sample: np.ndarray):
    """Calculates the value of the TSP solution for the given sample.

    Parameters:
        graph (nx.DiGraph): the graph of the TSP problem.
        sample (np.ndarray): the sample to evaluate.

    Returns:
        energy (float): the value of the solution.
    """
    N = len(graph.nodes)
    energy = 0.0
    solution_matrix = sample.reshape((N, N))
    solution_matrix[solution_matrix==-1] = 0
    if np.linalg.norm(solution_matrix.T@solution_matrix - np.eye(N)) != 0:
        return np.inf
    path = {time: city for city, time in enumerate(np.where(solution_matrix == 1)[1])}
    for time in range(N):
        city1 = path[time]
        if time == N-1:
            city2 = path[0]
        else:
            city2 = path[time+1]
        energy += graph[city1+1][city2+1]['weight']

    return energy
