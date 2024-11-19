import pathlib
import itertools

from ising.solvers.solver import Solver
from ising.model import BinaryQuadraticModel

class ExhaustiveSolver(Solver):
    def solve(self, bqm: BinaryQuadraticModel, log_file: pathlib.Path|None = None):
        with self.open_log(log_file, bqm) as log:
            min_energy = None
            solution = None
            for iteration, perm in enumerate(itertools.product([True, False], repeat=bqm.num_variables)):
                sample = {k: v for k,v in zip(bqm.linear.values(), perm)}
                energy = bqm.evaluate(sample)
                if energy < min_energy or min_energy is None:
                    min_energy = energy
                    solution = sample
                log.write(iteration, energy, sample)
        return solution, min_energy
