import numpy as np
from ising.solvers.solver import Solver
from ising.model import BinaryQuadraticModel
import pathlib
import random

class SCA(Solver):
    """Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the
    [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper

    Inherits from the abstract Solver base class.
    """

    def __init__(self, sigma:np.ndarray, file:pathlib.Path, S:int, T:float, r_t:float, q:float, r_q:float):
        """Initializes object.

        Args:
            sigma (np.ndarray): initial sample
            file (pathlib.Path): full path to the logger file
            S (int): total amount of iterations
            T (float): initial temperature
            r_t (float): temperature decrease rate
            q (float): initial penalty
            r_q (float): penalty increase rate
        """
        self.sigma = sigma
        self.S = S
        self.T = T
        self.r_t = r_t
        self.q = q
        self.r_q = r_q
        self.file = file
        self.hs = np.ndarray()

    def change_hyperparams(self)->None:
        """Changes hyperparameters according to update rule.
        """
        self.T = self.r_t*self.T
        self.q = self.r_q*self.q


    def solve(self, bqm:BinaryQuadraticModel, seed:int):
        N = bqm.num_variables
        h, J = bqm.to_ising()
        self.hs = np.copy(h)
        flipped_states = []
        random.seed(seed)
        with self.open_log(bqm) as log:
            for s in range(self.S):
                for x in range(N):
                    self.hs[x] += np.dot(J[x, :], self.sigma)
                    P = self.get_prob(x)
                    rand = random.random()
                    if P < rand:
                        flipped_states.append(x)
                for x in flipped_states:
                    self.change_node(x)
                self.change_hyperparams()
                flipped_states = []
                energy = bqm.eval(self.sigma)
                log.write(s, energy, self.sigma, self.T, self.q)

        return self.sigma, energy

    def get_prob(self, x:int):
        val = self.hs[x] * self.sigma[x] + self.q
        if -2 * self.T < val < 2 * self.T:
            return val / (4*self.T) + 0.5
        elif val > 2 * self.T:
            return 1.
        else:
            return 0.
