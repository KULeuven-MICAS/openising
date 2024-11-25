import numpy as np
import os

from ising.model.ising import IsingModel
import pathlib


def TSP(benchmark: pathlib.Path | str):
    folder = REPO_TOP + '/ising/generators/TSP_benchmark'
    for root, dirs, files in os.walk(folder):
        if benchmark not in files:
            raise OSError("benchmark not available")
    file = folder + "/" + benchmark
    file = pathlib.Path(file)
    last_weight_in_row = 0
    row = 0
    W = np.ndarray
    N = int
    weight_section = False
    with file.open() as f:
        for line in f:
            if line[:9] == "DIMENSION":
                N = int(line[10:])
                W = np.zeros((N, N))
            elif line[:19] == "EDGE_WEIGHT_SECTION":
                weight_section = True
            elif weight_section:
                line = list(line)
                part_row = []
                number = ""
                for i in range(len(line)):
                    if line[i] == " " and len(number) > 0:
                        part_row.append(int(number))
                        number = ""
                    elif line[i] != " ":
                        number += line[i]
                part_row.append(int(number))
                partial_row = np.array(part_row)
                n = partial_row.shape[0]
                W[row, last_weight_in_row : last_weight_in_row + n] = partial_row
                last_weight_in_row += n
                if last_weight_in_row >= N:
                    last_weight_in_row = 0
                    row += 1
                if row == N:
                    weight_section = False
    print("Weight matrix constructed")
    A = 8.
    B = 3.
    C = 2.
    J = np.zeros((N * N, N * N))
    h = np.zeros((N * N,))
    J, h = add_HA(J, h, W, N, A=A)
    J, h = add_HB(J, h, N, B=B)
    J, h = add_HC(J, h, N, C=C)
    J = 1 / 2 * (J + J.T)
    return IsingModel(J, h)


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
    return J, h


def add_HB(J:np.ndarray, h:np.ndarray, N:int, B:float):
    for time in range(N):
        for city1 in range(N):
            h[get_index(time, city1, N)] += (N - 2) / 2 * B
            for city2 in range(N):
                J[get_index(time, city1, N), get_index(time, city2, N)] += B / 4
    return J, h


def add_HC(J:np.ndarray, h:np.ndarray, N:int, C:float):
    for city in range(N):
        for time1 in range(N):
            h[get_index(time1, city, N)] += (N - 2) / 2 * C
            for time2 in range(N):
                J[get_index(time1, city, N), get_index(time2, city, N)] += C / 4
    return J, h
