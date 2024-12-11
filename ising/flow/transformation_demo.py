import numpy as np
from ising.model.ising import IsingModel

J = np.array([
    [0,  4, -3,  0,  5],
    [0,  0,  1, -4,  3],
    [0,  0,  0,  2, -1],
    [0,  0,  0,  0,  6],
    [0,  0,  0,  0,  0]
], dtype=float)

h = np.array([2, -1, 3, -2, 0])

model = IsingModel(J, h)
print(f"IsingModel:\n{model}")
print(f"Originally: mean={model.mean}, variance={model.variance}")

# Normalize
model.normalize()
print(model)
print(f"After normalization: mean={model.mean}, variance={model.variance}")

# Reconstruct
model.reconstruct()
print(model)
print(f"After reconstruction: mean={model.mean}, variance={model.variance}")
