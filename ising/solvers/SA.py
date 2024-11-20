from ising.solvers.solver import Solver#, SolverLogger
from ising.model import BinaryQuadraticModel
import numpy as np
import random
import pathlib
import math


class SA(Solver):
    """Performs simulated annealing (SA) as is seen in [Johnson et al.](https://faculty.washington.edu/aragon/pubs/annealing-pt1a.pdf)


    Inherits Solver abstract base class.
    """
    def __init__(
        self,
        sigma: np.ndarray,
        file: pathlib.Path,
        S: int,
        T: float,
        r_t: float,
    ):
        """ Initializes object.
        Args:
            sigma (np.ndarray): initial sample
            file (pathlib.Path): full path to file in which to log information
            S (int): amount of iterations
            T (float): initial temperature
            r_t (float): temperature decrease rate
        """
        self.sigma = sigma
        self.S = S
        self.T = T
        self.r_t = r_t
        self.file = file

    def set_T(self, T:float) -> None:
        """Changes temperature T

        Args:
            T (float): new temperature
        """
        self.T = T

    def change_node(self, node:int)->None:
        """Changes the spin of the node.

        Args:
            node (int): the node that needs to be changed. Should be in the range [0, N).
        """
        self.sigma[node] = -self.sigma[node]

    def solve(self, bqm:BinaryQuadraticModel, seed: int) -> tuple[np.ndarray, float]:
        """Performs the SA algorithm on the given model.

        Args:
            bqm (BinaryQuadraticModel): the model description that needs to be solved.
            seed (int): seed for generating random numbers. Needed for reproducing results.
        Returns:
            sigma (np.ndarray): the optimal solution
            cost_new (float): the optimal energy
        """
        N = bqm.num_variables
        random.seed(seed)
        with self.open_log(self.file, bqm) as log:
            for i in range(self.S):
                chosen_nodes = set()
                cost_old = bqm.eval(self.sigma)
                for j in range(N):
                    node = random.choice(list(set(range(N)) - chosen_nodes))
                    self.change_node(node)
                    cost_new = bqm.eval(self.sigma)
                    delta = cost_new - cost_old
                    P = delta / self.T
                    rand = -math.log(random.random())
                    if delta < 0 or P < rand:
                        cost_old = cost_new
                    else:
                        self.change_node(node)
                    chosen_nodes.add(node)
                self.set_T(self.r_t * self.T)
                log.write(i, cost_new, self.sigma)
        return self.sigma, cost_new
