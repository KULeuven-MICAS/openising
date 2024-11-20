from ising.solvers.solver import Solver
from ising.model import BinaryQuadraticModel
import numpy as np
import pathlib

class BRIM(Solver):
    def __init__(self, v, S, file:pathlib.Path, dt, kmin, kmax, C, G):
        self.v = v
        self.sigma = np.sign(v)
        self.S = S
        self.dt = dt
        self.kmin = kmin
        self.kmax = kmax
        self.C = C
        self.G = G
        self.cycle_duration = S*dt/10
        self.file = file

    def k(self, t):
        return self.kmax if int(t // (self.cycle_duration/2)) % 2 == 0 else self.kmin

    def set_sigma(self):
        self.sigma = np.sign(self.v)

    def solve(self, bqm:BinaryQuadraticModel):
        N = bqm.num_variables
        h, J = bqm.to_ising()
        tk = 0.
        def dvdt(t, v):
            V = np.array([v]*N)
            dv = 1/self.C*(self.G*np.tanh(self.k(t)*np.tanh(self.k(t)*v)) - self.G*v - np.sum(J*(V - V.T), 0))
            dv = np.where(np.all(np.array([dv > 0., v >=  1.]), 0), np.zeros((N,)), dv)
            dv = np.where(np.all(np.array([dv < 0., v <= -1.]), 0), np.zeros((N,)), dv)
            return dv
        with self.open_log(self.file, bqm) as log:
            for i in range(self.S):
                k1 = dvdt(tk, self.v)
                k2 = dvdt(tk + self.dt/2, self.v + self.dt/2*k1)
                k3 = dvdt(tk + self.dt/2, self.v + self.dt/2*k2)
                k4 = dvdt(tk + self.dt, self.v + self.dt*k3)

                self.v += self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
                self.set_sigma()
                energy = bqm.eval(self.sigma)
                tk += self.dt
                log.write(tk, energy, self.sigma, self.v)
        return self.sigma, energy
