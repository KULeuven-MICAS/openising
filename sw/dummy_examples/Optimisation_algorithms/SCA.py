import random
import numpy as np
import math

def SCA(s_init, J, h, S, q_init, T_init, r_q, r_t):
    q = q_init
    T = T_init
    N = np.length(s_init)
    sigma = s_init
    tau = sigma
    for s in range(S):
        for x in range(N):
            h[x] = np.inner(J[x,:], sigma) + h[x]
            P = get_prob(T, h[x], q, sigma[x])
            rand = random.randrange(0, 1)
            if P < rand:
                tau[x] = -sigma[x]
        for x in range(N):
            sigma[x] = tau[x]
        q = q*r_q
        T = T*r_t
    return sigma


def get_prob(Temp, hx, qs, sigmax):
   x = 1/Temp*(hx*sigmax + qs)/2
   return 1/(1+math.exp(x))