import numpy as np
from ising.model.ising import IsingModel
import os
import pathlib

def MaxCut(benchmark:pathlib.Path|str):
    folder = REPO_TOP + '.ising/generators/Max-Cut_benchmarks'
    file = folder + benchmark
    if file not in os.walk(folder)[2]:
        raise OSError('benchmark is not available')
    data = np.genfromtxt(file, delimiter=' ')
    N = int(data[0, 0])
    J = np.zeros((N, N))
    h = np.zeros((N,))
    for row in data:
        i, j, weight = int(row[0])-1, int(row[1])-1, row[2]
        J[i, j] = -weight/2
        J[j, i] = -weight/2
    return IsingModel(J, h)
