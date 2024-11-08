import numpy as np
import networkx as nx
import math
import random
import helper_functions as hf

def SA(T:float, r_T:float, S:int, J:np.ndarray, h:np.ndarray, sigma:np.ndarray, verbose:bool=False) -> tuple[np.ndarray, list]:
    """
    Performs simulated annealing (SA) as is seen in https://faculty.washington.edu/aragon/pubs/annealing-pt1a.pdf

    :param float T: initial temperature, should be high enough in the beginning to allow random flipping
    :param float r_T: temperature ratio in (0, 1). Ratio by which the temperature decreases every iteration
    :param int S: maximum number of iterations
    :param np.ndarray J: interaction coefficients of the problem
    :param np.ndarray h: self-interaction coefficients of the problem
    :param np.ndarray sigma: initial solution to the problem
    
    :return sigma (np.ndarray): optimal solution
    :return energies (list): energy values during optimization algorithm
    """
    #TODO: change representation of model to own representation
    N = np.shape(sigma)[0]
    sigma_new = np.copy(sigma)
    energies = []
    if verbose:
        header = ['Iteration count', 'Energy']
        print("{: >20} {: >20} ".format(*header))
    for i in range(S):
        for node in range(N):
            sigma_new[node] = -sigma[node]
            cost_new = hf.compute_energy(J, h, sigma_new)
            cost_old = hf.compute_energy(J, h, sigma)
            delta = cost_new - cost_old
            P = delta/T
            rand = -math.log(random.random())
            if delta < 0:
                sigma[node] = sigma_new[node]
            elif P < rand:
                sigma[node] = sigma_new[node]
            else:
                sigma_new[node] = sigma[node]
        cost = hf.compute_energy(J, h, sigma)
        energies.append(cost)
        if verbose:
            row = [i, str(cost)]
            print("{: >20} {: >20}".format(*row))
        T = r_T*T
        sigma_new = np.copy(sigma)
    return sigma, energies
