import numpy as np
import openjij as oj
import random


def problem_gen(N:int, Jmin:float, Jmax:float, hmin:float, hmax:float): 
    """
    Creates an Ising problem of size N. The problem is fully-connected and creates weights between the given ranges.

    Args:
        N: the size of the problem
        Jmin, Jmax: the range of the interaction weights
        hmin, hmax: the range of the magnetic field weights

    Output:
        J: a numpy array with the interaction weights
        h: a numpy array with the magnetic field weights
        optim: tuple of the openjij optimal solution and corresponding energy
    """
    J = np.zeros((N,N))
    h = np.zeros((N,))
    ojmat = np.array((N,N))
    for i in range(N):
        for j in range(N):
            if i==j:
                h[i] = random.uniform(hmin, hmax)
                ojmat[i,j] = h[i]
            else:
                J[i,j] = random.uniform(Jmin, Jmax)
                ojmat[i,j] = J[i,j]

    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(ojmat, vartype='SPIN')
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=100)
    print(f"Optimal state: {response.first.sample}")
    print(f"Optimal energy: {response.first.energy}")
    return J, h, (response.first.sample, response.first.energy)
    