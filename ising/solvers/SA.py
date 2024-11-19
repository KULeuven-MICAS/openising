from ising.solvers.solver import Solver
import numpy as np
import random
import math

class SA(Solver):
    def __init__(self, sigma, model, file, S, T, r_t, verbose):
        Solver.__init__(sigma, model, file, verbose)
        self.S = S
        self.T = T
        self.r_t = r_t

    def set_T(self, T):
        self.T = T

    def change_node(self, node):
        self.sigma[node] = -self.sigma[node]

    def run(self, seed) -> tuple[np.ndarray, float]:
        N = self.model.num_variables
        random.seed(seed)
        if self.verbose:
            self.print(header=True)
        for i in range(self.S):
            chosen_nodes = set()
            for j in range(N):
                node = random.choice(list(set(range(N)) - chosen_nodes))
                cost_old = self.get_energy()
                self.change_node(node)
                cost_new = self.compute_energy()
                delta = cost_new - cost_old
                P = delta / self.T
                rand = -math.log(random.random())
                if delta < 0 or P < rand:
                    self.set_energy(cost_new)
                else:
                    self.change_node(node)
                chosen_nodes.add(node)
            if self.verbose:
                self.print(header=False, it=i)
            self.set_T(self.r_t*self.T)
