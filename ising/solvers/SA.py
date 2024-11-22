from ising.solvers.solver import Solver
from ising.model.ising import IsingModel
import numpy as np
import random
import pathlib
import math


class SA(Solver):
    """Performs simulated annealing (SA) as is seen in [Johnson et al.](https://faculty.washington.edu/aragon/pubs/annealing-pt1a.pdf)


    Inherits Solver abstract base class.
    """

    def solve(
        self, model: IsingModel, file: pathlib.Path, seed: int, sample: np.ndarray, S: int, T: float, r_t: float
    ) -> tuple[np.ndarray, float]:
        """Performs the SA algorithm on the given model.

        Args:
            bqm (BinaryQuadraticModel): the model description that needs to be solved.
            seed (int): seed for generating random numbers. Needed for reproducing results.
        Returns:
            sigma (np.ndarray): the optimal solution
            cost_new (float): the optimal energy
        """
        N = model.num_variables
        random.seed(seed)
        with self.open_log(file, model) as log:
            for i in range(S):
                chosen_nodes = set()
                cost_old = model.evaluate(sample)
                for j in range(N):
                    node = random.choice(list(set(range(N)) - chosen_nodes))
                    sample = self.change_node(sample, node)
                    cost_new = model.evaluate(sample)
                    delta = cost_new - cost_old
                    P = delta / T
                    rand = -math.log(random.random())
                    if delta < 0 or P < rand:
                        cost_old = cost_new
                    else:
                        self.change_node(node)
                    chosen_nodes.add(node)
                T = r_t * T
                log.write(i, cost_new, sample)
        return sample, cost_new
