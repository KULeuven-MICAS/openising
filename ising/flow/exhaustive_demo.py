import os
import pathlib
import time
import h5py
import numpy as np

import ising.utils.adj as adj
import ising.generators as gen
from ising.solvers import ExhaustiveSolver

file = pathlib.Path(os.environ['TOP']) / 'ising/flow/logs/exhaustive_demo.log'
problem_size = 14

adj_matrix = adj.complete(problem_size)
linear = np.ones(problem_size)
model = gen.randint(adj_matrix, linear, low=-20, high=20)
print("Generated the following model:\n", model)

start_time = time.perf_counter()
state, energy = ExhaustiveSolver().solve(model, file=file)
end_time = time.perf_counter()

print(f"The following solution was found in {end_time - start_time} seconds:")
print(f"  state={state}, energy={energy}")

file = h5py.File(file)
print("Our log contained the following fields:")
for field, dset in file.items():
    print(f"  {field:7}- shape={dset.shape}")
print("And the following attributes:")
for attr, val in file.attrs.items():
    print(f"  {attr:16}- {val}")
