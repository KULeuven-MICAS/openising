import pathlib
import itertools
import numpy as np

from ising.model.ising import IsingModel
from ising.solvers.solver import Solver

class ExhaustiveSolver(Solver):
    def solve(self, model: IsingModel, log_file: pathlib.Path|None = None):
        with self.open_log(log_file, model) as log:
            min_energy = None
            solution = None
            sample = np.ones(model.num_variables, dtype=float)
            for iteration, perm in enumerate(itertools.product([-1, 1], repeat=model.num_variables)):
                sample[:] = perm
                energy = model.evaluate(sample)
                if min_energy is None or energy < min_energy:
                    min_energy = energy
                    solution = sample.copy()
                log.write(iteration, energy, sample)
        return solution, min_energy
