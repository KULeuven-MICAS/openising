import numpy as np
import networkx as nx

from ising.model.ising import IsingModel

__all__ = ["TSP"]

def TSP(graph: nx.DiGraph, A: float, B: float, C: float) -> IsingModel:
    """Generates an Ising model of the TSP from the given directed graph

    Args:
        graph (nx.DiGraph): graph on which the TSP problem will be solved
        A (float): weight constant
        B (float): place constraint constant
        C (float): time constraint constant

    Returns:
        model (IsingModel): Ising model of the TSP
    """
    N = len(graph.nodes)
    W = np.zeros((N, N))
    for city1 in range(N):
        for city2 in range(N):
            if graph.has_edge(city1, city2):
                W[city1, city2] = graph[city1][city2]["weight"]
    J = np.zeros((N * N, N * N))
    h = np.zeros((N * N,))
    add_HA(J, h, W, N, A=A)
    add_HB(J, h, N, B=B)
    add_HC(J, h, N, C=C)
    J = 1 / 2 * (J + J.T)
    J = np.triu(J)
    c = (N*N)*(B+C)/4
    return IsingModel(-J, -h, -c)


def get_index(time:int, city:int, N:int):
    if time >= N:
        time -= N
    return (city * N) + time


def add_HA(J:np.ndarray, h:np.ndarray, W:np.ndarray, N:int, A:float):
    for city1 in range(N):
        for city2 in range(N):
            for time in range(N):
                if city1 != city2:
                    J[get_index(time, city1, N), get_index(time + 1, city2, N)] += A / 2 * W[city1, city2]
                    h[get_index(time, city1, N)] += A / 2 * W[city1, city2]


def add_HB(J:np.ndarray, h:np.ndarray, N:int, B:float):
    for time in range(N):
        for city1 in range(N):
            h[get_index(time, city1, N)] += (N - 2) / 2 * B
            for city2 in range(N):
                if city1 != city2:
                    J[get_index(time, city1, N), get_index(time, city2, N)] += B / 4


def add_HC(J:np.ndarray, h:np.ndarray, N:int, C:float):
    for city in range(N):
        for time1 in range(N):
            h[get_index(time1, city, N)] += (N - 2) / 2 * C
            for time2 in range(N):
                if time1 != time2:
                    J[get_index(time1, city, N), get_index(time2, city, N)] += C / 4
