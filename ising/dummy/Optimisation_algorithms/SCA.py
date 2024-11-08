import random
import numpy as np
import math
import helper_functions as hf

def SCA(s_init:np.ndarray, J:np.ndarray, h_init:np.ndarray, S:int, q:float, T:float, r_q:float, r_t:float, verbose:bool=False)->tuple[np.ndarray, list]:
    """
    Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper 
    
    :param np.ndarray s_init: initial spin configuration
    :param np.ndarray J: interaction coefficients
    :param np.ndarray h_init: initial local magnetic field coefficients
    :param float q: initial value of penalty hyperparameter
    :param float T: initial temperature
    :param float r_q: penalty increase rate
    :param float r_t: temperature decrease rate
    :param bool verbose: whether or not to show information during running
    :return sigma (np.ndarray): optimal spin configuration
    :return energy_list (list): list of all the energies during run
    """
    N = np.shape(s_init)[0]
    sigma = s_init
    h = np.copy(h_init)
    flipped_states = []
    energy_list = []
    if verbose:
        header = ['Iteration Count', 'Energy']
        print("{: >20} {: >20}".format(*header))
    for s in range(S):
        for x in range(N):
            if s==0:
                h[x] = np.inner(J[x, :], sigma) + h[x]
            else:
                h[x] = 2*np.inner(J[x, :], sigma) + h[x]
            P = get_prob(T, h[x], q, sigma[x])
            rand = random.random()
            if P < rand:
                flipped_states.append(x)
        for x in flipped_states:
            sigma[x] = -sigma[x]
        q = q*r_q
        T = T*r_t
        flipped_states = []
        energy = hf.compute_energy(J, h_init, sigma)
        if verbose:
            row = [s, str(energy)]
            print("{: >20} {: >20}".format(*row))
        energy_list.append(energy)

    return sigma, energy_list


def get_prob(Temp, hx, qs, sigmax):
   val = hx*sigmax + qs
   if -2*Temp < val < 2*Temp:
       return val/(4*Temp) + 0.5
   elif val > 2*Temp:
       return 1.
   else:
       return 0.
    