import os
import pathlib
import ising.utils.adj as adj
import ising.generators as gen
from ising.solvers.exhaustive import ExhaustiveSolver
#import numpy as np

log_file = pathlib.Path(os.environ['TOP']) / 'ising/flow/logs/exhaustive_solver.log'

model = gen.randint(adj.complete(16), low=-5, high=5)
print(model)

solution, min_energy = ExhaustiveSolver().solve(model, log_file=log_file)
print(solution, min_energy)
