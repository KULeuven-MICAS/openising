import random
import numpy as np
import math

def SCA(s_init, J, h_init, S, q_init, T_init, r_q, r_t):
    q = q_init
    T = T_init
    N = np.shape(s_init)[0]
    sigma = s_init
    tau = np.copy(s_init)
    h = np.copy(h_init)
    energy_list = []
    header = ['Iteration Count', 'Energy']
    print("{: >20} {: >20}".format(*header))
    for s in range(S):
        for x in range(N):
            h[x] = np.inner(J[x,:], sigma) + h[x]
            P = get_prob(T, h[x], q, sigma[x])
            rand = random.random()
            if P < rand:
                tau[x] = -sigma[x]
        for x in range(N):
            sigma[x] = tau[x]
        q = q*r_q
        T = T*r_t
        energy = compute_energy(J, h_init, sigma)
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


def compute_energy(J, h, sigma):
    return -np.inner(sigma.T, np.inner(J, sigma)) - np.inner(h.T, sigma)
    