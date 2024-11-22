import numpy as np
from ising.solvers.solver import Solver
from ising.model.ising import IsingModel
import pathlib
import random


class SCA(Solver):
    """Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the
    [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper

    Inherits from the abstract Solver base class.
    """
    def change_hyperparam(self, param: float, rate: float) -> float:
        """Changes hyperparameters according to update rule."""
        return param * rate

    def solve(
        self,
        model: IsingModel,
        file: pathlib.Path,
        seed: int,
        sample: np.ndarray,
        S: int,
        T: float,
        r_t: float,
        q: float,
        r_q: float,
    ):
        N = model.num_variables
        hs = np.copy(model.h)
        flipped_states = []
        random.seed(seed)
        with self.open_log(file, model) as log:
            for s in range(S):
                for x in range(N):
                    hs[x] += np.dot(model.J[x, :], sample)
                    P = self.get_prob(hs[x], sample[x], q, T)
                    rand = random.random()
                    if P < rand:
                        flipped_states.append(x)
                for x in flipped_states:
                    sample = self.change_node(sample, x)
                T = self.change_hyperparam(T, r_t)
                q = self.change_hyperparam(q, r_q)
                flipped_states = []
                energy = model.evaluate(sample)
                log.write(s, energy, sample)

        return self.sigma, energy

    def get_prob(self, hsx, samplex, q, T):
        val = hsx * samplex + q
        if -2 * T < val < 2 * T:
            return val / (4 * T) + 0.5
        elif val > 2 * T:
            return 1.0
        else:
            return 0.0
