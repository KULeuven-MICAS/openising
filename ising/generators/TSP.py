import numpy as np
import networkx as nx
import time as t

from ising.model.ising import IsingModel

__all__ = ["TSP"]


def ATSP(graph: nx.DiGraph, A: float = 1.0, B: float = 5.0, C: float = 5.0) -> IsingModel:
    """Generates an Ising model of the assymetric TSP from the given directed graph

    Args:
        graph (nx.DiGraph): graph on which the TSP problem will be solved
        A (float, optional): weight constant. Defaults to 1.
        B (float, optional): place constraint constant. Defaults to 5.
        C (float, optional): time constraint constant. Defaults to 5.

    Returns:
        model (IsingModel): Ising model of the TSP
    """
    N = len(graph.nodes)
    W = np.zeros((N, N))
    for city1 in range(N):
        for city2 in range(N):
            if graph.has_edge(city1, city2):
                W[city1, city2] = graph[city1][city2]["weight"]

    # Make a QUBO representation of the TSP problem
    Q = np.zeros((N**2, N**2))
    add_HA(Q, W, N, A=A)
    add_HB(Q, N, B=B)
    add_HC(Q, N, C=C)
    return IsingModel().from_qubo(Q)

def TSP(graph: nx.DiGraph, A: float=1., B:float=5., C:float=5.):
    """Generates an Ising model of the standard TSP from the given graph.

    Args:
        graph (nx.DiGraph): The graph representing the TSP problem.
        A (float, optional): The weight constant. Defaults to 1..
        B (float, optional): The place constraint constant. Defaults to 5..
        C (float, optional): The time constraint constant. Defaults to 5..
    """

def generate_random_TSP(
    N: int, seed: int = 0, weight_constant: float = 1.0, time_constraint: float = 5.0, place_constraints: float = 5.0
) -> tuple[IsingModel, nx.DiGraph]:
    if seed == 0:
        seed = int(t.time())
    np.random.seed(seed)
    W = np.random.choice(10, (N, N))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and W[i, j] != 0:
                graph.add_edge(i, j, weight=W[i, j])
    model = TSP(graph, A=weight_constant, B=place_constraints, C=time_constraint)
    return model, graph


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


def add_HA(Q: np.ndarray, W: np.ndarray, N: int, A: float):
    """Generates the objective function term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        W (np.ndarray): the weight matrix
        N (int): the amount of cities.
        A (float): the objective function constant.
    """
    for city1 in range(N):
        for city2 in range(N):
            for time in range(N):
                index1 = get_index(time, city1, N)
                index2 = get_index(time, city2, N)
                Q[index1, index2] += A * W[city1, city2]


def add_HB(Q: np.ndarray, N: int, B: float):
    """Generates the time constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities
        B (float): the time constraint constant.
    """
    for time in range(N):
        for city1 in range(N):
            index1 = get_index(time, city1, N)
            Q[index1, index1] += -B
            for city2 in range(city1):
                index2 = get_index(time, city2, N)
                Q[index1, index2] += 2 * B


def add_HC(Q: np.ndarray, N: int, C: float):
    """Generates the place constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities.
        C (float): the place constraint constant.
    """
    for city in range(N):
        for time1 in range(N):
            index1 = get_index(time1, city, N)
            Q[index1, index1] += -C
            for time2 in range(time1):
                index2 = get_index(time2, city, N)
                Q[index1, index2] += 2 *C


def get_TSP_value(graph: nx.DiGraph, sample: np.ndarray):
    """Calculates the value of the TSP solution for the given sample.

    Parameters:
        graph (nx.DiGraph): the graph of the TSP problem.
        sample (np.ndarray): the sample to evaluate.

    Returns:
        energy (float): the value of the solution.
    """
    N = len(graph.nodes)
    path = [-1] * N
    for city in range(N):
        for time in range(N):
            if sample[get_index(time, city, N)] == 1:
                path[time] = city
                break
    energy = 0.0
    for i in range(N):
        city1 = path[i]
        city2 = path[(i + 1) % N]  # wrap around to the first city
        if graph.has_edge(city1, city2):
            energy += graph[city1][city2]["weight"]

    return energy
