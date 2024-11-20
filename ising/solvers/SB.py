from ising.model import BinaryQuadraticModel
from ising.solvers.solver import Solver
import numpy as np
import pathlib

class SB(Solver):
    """Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    This class inherits from the abstract Solver base class.
    """

    def __init__(self, x:np.ndarray, y:np.ndarray, file:pathlib.Path, S:int, a0:float, c0:float, dt:float, at:callable):
        self.x = x
        self.sigma = np.sign(x)
        self.y = y
        self.S = S
        self.a0 = a0
        self.c0 = c0
        self.dt = dt
        self.at = at
        self.file = file

    def update_x(self, node):
        self.x[node] += self.a0*self.y[node]*self.dt

    def set_sigma(self):
        self.sigma = np.sign(self.x)

    def update_rule(self, node):
        self.x[node] = np.sign(self.x[node])
        self.y[node] = 0.

    def solve(self, bqm:BinaryQuadraticModel):
        pass

class ballisticSB(SB):

    def __init__(self, x, y, file, S, a0, c0, dt, at):
        super().__init__(x, y, file, S, a0, c0, dt, at)

    def solve(self, bqm:BinaryQuadraticModel):
        tk = 0
        N = bqm.num_variables
        h, J = bqm.to_ising()
        with self.open_log(self.file, bqm) as log:
            for i in range(self.S):
                for j in range(N):
                    self.y[j] += (-(self.a0 - self.at(tk))*self.x[j] + self.c0*np.dot(J[:, j], self.x) + \
                             self.c0*h[j])*self.dt
                    self.update_x(j)
                    if np.abs(self.x[j]) > 1:
                        self.update_rule(j)
                self.set_sigma()
                energy = bqm.eval(self.sigma)
                log.write(tk, energy, self.sigma, self.x, self.y, self.at(tk))
                tk += self.dt
        return self.sigma, energy


class discreteSB(SB):

    def __init__(self, x, y, file, S, a0, c0, dt, at):
        super().__init__(x, y, file, S, a0, c0, dt, at)

    def solve(self, bqm):
        h, J = bqm.to_ising()
        N = bqm.num_variables
        tk = 0.
        with self.open_log(self.file, bqm) as log:
            for i in range(self.S):
                for j in range(N):
                    self.y[j] += (-(self.a0 - self.at(tk))*self.x[j] + self.c0*np.inner(J[:, j], np.sign(self.x)) + \
                            self.c0*h[j])*self.dt
                    self.update_x()
                    if np.abs(self.x[j]) > 1:
                        self.update_rule(j)
                self.set_sigma()
                energy = bqm.eval(self.sigma)
                tk += self.dt
                log.write(tk, energy, self.sigma, self.x, self.y, self.at(tk))
        return self.sigma
