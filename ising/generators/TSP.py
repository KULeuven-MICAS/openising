import numpy as np
import networkx as nx
import time as t
import os
import pathlib

from ising.model.ising import IsingModel


__all__ = ["TSP"]


def TSP(graph: nx.DiGraph, weight_constant: float = 1.0) -> IsingModel:
    """Generates an Ising model of the assymetric TSP from the given directed graph

    Args:
        graph (nx.DiGraph): graph on which the TSP problem will be solved
        weight constant (float, optional): weight constant of the original objective function. Defaults to 1.

    Returns:
        model (IsingModel): Ising model of the TSP
    """
    N = len(graph.nodes)
    W = nx.linalg.adjacency_matrix(graph).toarray()
    maxW = np.max(W)
    B = weight_constant * maxW
    C = weight_constant * maxW

    # Add pseudo weights for non-existing connections
    for i in range(N):
        for j in range(N):
            if i != j and W[i, j] == 0:
                W[i,j] = 10*maxW

    # Make a QUBO representation of the TSP problem
    J = np.zeros((N**2, N**2))
    h = np.zeros(N**2)
    add_HA(J, h, W, N, A=weight_constant)
    add_HB(J, h, N, B=B)
    add_HC(J, h, N, C=C)
    # Q = np.triu(Q, k=0)
    print(np.linalg.cond(J))
    J = np.triu(J, k=1)
    return IsingModel(J, h)


def generate_random_TSP(
    N: int, seed: int = 0, weight_constant: float = 1.0, time_constraint: float = 5.0, place_constraints: float = 5.0
) -> tuple[IsingModel, nx.DiGraph]:
    if seed == 0:
        seed = int(t.time())
    dummy_problem = pathlib.Path(os.getenv("TOP")) / pathlib.Path(f"ising/benchmarks/TSP_dummy/N{N}_dummy.txt")
    if dummy_problem.exists():
        W = np.zeros((N, N))
        first_line = True
        with dummy_problem.open() as file:
            for line in file:
                if not first_line:
                    line_list = line.split()
                    W[int(line_list[0]) - 1, int(line_list[1]) - 1] = float(line_list[2])
                first_line = False
    else:
        np.random.seed(seed)
        W = np.random.choice(10, (N, N))
        W = (W + W.T) / 2
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, N + 1))
    for i in range(N):
        for j in range(N):
            if i != j:
                if W[i, j] != 0:
                    graph.add_edge(i + 1, j + 1, weight=W[i, j])
    model = TSP(graph, weight_constant=weight_constant)
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


def add_HA(J: np.ndarray, h: np.ndarray, W: np.ndarray, N: int, A: float):
    """Generates the objective function term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        W (np.ndarray): the weight matrix
        N (int): the amount of cities.
        A (float): the objective function constant.
    """

    for city1 in range(N):
        for time1 in range(N):
            index1 = get_index(time1, city1, N)
            for city2 in range(N):
                if city1 != city2:
                    h[index1] -= A / 2 * W[city1, city2]
                for time2 in range(N):
                    if (
                        time2 == (time1 + 1)%N
                        # or (time1 == 0 and time2 == N - 1)
                        # or (time1 == N - 1 and time2 == 0)
                        or time1 == (time2 + 1) % N
                    ) and city1 != city2:
                        index2 = get_index(time2, city2, N)
                        J[index1, index2] -= A / 8 * W[city1, city2]

    # for city1 in range(N):
    #     for city2 in range(N):
    #         for time in range(N):
    #             index1 = get_index(time, city1, N)
    #             index2 = get_index(time + 1, city2, N)
    #             Q[index1, index2] += A * W[city1, city2]


def add_HB(J: np.ndarray, h: np.ndarray, N: int, B: float):
    """Generates the time constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities
        B (float): the time constraint constant.
    """
    for time1 in range(N):
        for city1 in range(N):
            index1 = get_index(time1, city1, N)
            h[index1] -= (N - 2) * B / 2
            for city2 in range(N):
                for time2 in range(N):
                    index2 = get_index(time2, city2, N)
                    if time1 == time2 and city1 != city2:
                        J[index1, index2] -= B / 4
                    elif city1 == city2 and time1 == time2:
                        J[index1, index2] -= B / 4
                    # Q[index2, index1] += B *2


def add_HC(J: np.ndarray, h: np.ndarray, N: int, C: float):
    """Generates the place constraint term for the transformed TSP problem in QUBO formulation.

    Args:
        Q (np.ndarray): the current QUBO matrix.
        N (int): the amount of cities.
        C (float): the place constraint constant.
    """
    for city1 in range(N):
        for time1 in range(N):
            index1 = get_index(time1, city1, N)
            h[index1] -= (N - 2) * C / 2
            for city2 in range(N):
                for time2 in range(N):
                    index2 = get_index(time2, city2, N)
                    if (city1 == city2) and (time1 != time2):
                        J[index1, index2] -= C / 4
                    elif (city1 == city2) and (time1 == time2):
                        J[index1, index2] -= C / 4
                    # Q[index1, index2] += C * 2
                    # Q[index2, index1] += C * 2


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
    for city1 in range(N):
        for city2 in range(N):
            if city1 != city2:
                for time in range(N):
                    index1 = get_index(time, city1, N)
                    index2 = get_index(time + 1, city2, N)
                    if sample[index1] == 1 and sample[index2] == 1 and graph.has_edge(city1 + 1, city2 + 1):
                        energy += graph[city1 + 1][city2 + 1]["weight"]

    return energy
