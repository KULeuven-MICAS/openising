import random
import numpy as np
import math

def SCA(s_init, J, h_init, S, q_init, T_init, r_q, r_t):
    q = q_init
    T = T_init
    N = np.shape(s_init)[0]
    sigma = s_init
    tau = sigma
    h = np.copy(h_init)
    energy_list = []
    header = ['Iteration Count', 'Energy', 'current state']
    print("{: >20} {: >20} {: >20}".format(*header))
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
        energy = compute_energy(J, h_init, q, sigma, tau)
        row = [s, str(energy), str(sigma)]
        print("{: >20} {: >20} {: >20}".format(*row))
        energy_list.append(energy)

    return sigma, energy_list


def get_prob(Temp, hx, qs, sigmax):
   x = 1/Temp*(hx*sigmax + qs)/2
   return 1/(1+math.exp(x))


def compute_energy(J, h, q, sigma, tau):
    energy = 0.
    N = np.shape(sigma)[0]
    for i in range(N):
        energy += h[i]*sigma[i]
        for j in range(i+1, N):
            energy += J[i, j]*sigma[i]*sigma[j]
    return energy