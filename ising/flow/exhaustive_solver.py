import pathlib
import ising.utils.adj as adj
import ising.generators as gen
from ising.solvers.exhaustive import ExhaustiveSolver

log_file = pathlib.Path.home() / 'projects/ising/ising/flow/logs/exhaustive_solver.log'

model = gen.randint(adj.complete(16), low=-5, high=5)
print(model)

solution, min_energy = ExhaustiveSolver().solve(model, log_file=log_file)
print(solution, min_energy)
