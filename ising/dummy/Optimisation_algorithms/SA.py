import numpy as np
import networkx as nx
import math
import random

def SA(T, r_T, S, J, h, sigma):
    """
    Performs simulated annealing (SA) as is seen in https://faculty.washington.edu/aragon/pubs/annealing-pt1a.pdf

    :param T: initial temperature, should be high enough in the beginning to allow random flipping
    :param r_T: temperature ratio in (0, 1). Ratio by which the temperature decreases every iteration
    :param T_min: minimum temperature to stop algorirthm
    :param J: interaction coefficients of the problem
    :param h: self-interaction coefficients of the problem
    :param sigma: initial solution to the problem
    
    :return sigma: optimal solution
    :return energies: energy values during optimization algorithm
    """
    N = np.shape(sigma)[0]
    sigma_new = np.copy(sigma)
    energies = []
    header = ['Iteration count', 'Energy']
    print("{: >20} {: >20} ".format(*header))
    for i in range(S):
        for node in range(N):
            sigma_new[node] = -sigma[node]
            cost_new = -np.inner(sigma_new.T, np.inner(J, sigma_new)) - np.inner(h.T, sigma_new)
            cost_old = -np.inner(sigma.T, np.inner(J, sigma)) - np.inner(h.T, sigma)
            delta = cost_new - cost_old
            P = delta/T
            rand = -math.log(random.random())
            if delta < 0:
                sigma[node] = sigma_new[node]
            elif P < rand:
                sigma[node] = sigma_new[node]
            else:
                sigma_new[node] = sigma[node]
        cost = -np.inner(sigma.T, np.inner(J, sigma)) - np.inner(h.T, sigma)
        energies.append(cost)
        row = [i, str(cost)]
        print("{: >20} {: >20}".format(*row))
        T = r_T*T
        sigma_new = np.copy(sigma)
    return sigma, energies
