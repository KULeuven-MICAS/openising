import numpy as np
from ising.model.ising import IsingModel
from ising.solvers.SB import ballisticSB

J = np.array([
    [0,  4, -3,  0,  5],
    [0,  0,  1, -4,  3],
    [0,  0,  0,  2, -1],
    [0,  0,  0,  0,  6],
    [0,  0,  0,  0,  0]
], dtype=float)

h = np.array([2, -1, 3, -2, 0])

dt = 0.25
c0 = 1.
a0 = 1.
num_iter = 1000

model = IsingModel(J, h)
x = np.random.choice([-1, 1] , (model.num_variables,)) * 0.1
y = np.zeros_like(x)

optim_state, optim_energy = ballisticSB().solve(model, x, y, num_iter, c0, dt, a0)
print(f"IsingModel:\n{model}")
print(f"Originally: mean={model.mean}, variance={model.variance}")
print(f"{optim_energy=} and {optim_state=}")

# Normalize
model.normalize()
optim_state, optim_energy = ballisticSB().solve(model, x, y, num_iter, c0, dt, a0)
print(model)
print(f"After normalization: mean={model.mean}, variance={model.variance}")
print(f"{optim_energy=} and {optim_state=}")


# Reconstruct
model.reconstruct()
optim_state, optim_energy = ballisticSB().solve(model, x, y, num_iter, c0, dt, a0)
print(model)
print(f"After reconstruction: mean={model.mean}, variance={model.variance}")
print(f"{optim_energy=} and {optim_state=}")
