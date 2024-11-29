import os
import pathlib
import time
import numpy as np

import ising.utils.adj as adj
import ising.generators as gen
from ising.solvers import DSASolver, SASolver, ExhaustiveSolver

file = pathlib.Path(os.environ['TOP']) / 'ising/flow/logs/simulated_annealing_demo.log'
problem_size = 18

adj_matrix = adj.complete(problem_size)
linear = np.ones(problem_size)
model = gen.randint(adj_matrix, linear, low=-20, high=20)
print("Generated the following model:\n", model)

initial_state = np.random.choice([-1, 1], size=problem_size).astype(np.int8)
num_iterations = 10000
initial_temp = 100
cooling_rate = 0.999
dsa_cooling_rate = 0.999

start_time = time.perf_counter()
state, energy = SASolver().solve(model, initial_state, num_iterations, initial_temp, cooling_rate, file=file)
end_time = time.perf_counter()

print(f"SA: The following solution was found in {end_time - start_time} seconds:")
print(f"  state={state}, energy={energy}")

start_time = time.perf_counter()
state, energy = DSASolver().solve(model, initial_state, num_iterations, initial_temp, dsa_cooling_rate, file=file)
end_time = time.perf_counter()

print(f"DSA: The following solution was found in {end_time - start_time} seconds:")
print(f"  state={state}, energy={energy}")

optimal_state, optimal_energy = ExhaustiveSolver().solve(model)
print("And the optimal solution is:")
print(f"  state={optimal_state}, energy={optimal_energy}")
