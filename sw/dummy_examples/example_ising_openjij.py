import openjij as oj
import numpy as np
import random
import matplotlib.pyplot as plt

# Create interaction coefficients and longitudinal magnetic fields
# OpenJij accepts problems in dictionary type
print("------------------------------------------")
print("Original Example")
print("------------------------------------------")
N = 5
h = {i: -1 for i in range(N)}
J = {(i,j): -1 for i in range(N) for j in range(i+1,N)}

print('h_i: ', h)
print('Jij: ', J)

# 1. Create instance of Sampler to solve problem
sampler = oj.SASampler()
# 2. Solve problem(h, J) by using method of sampler
response = sampler.sample_ising(h=h, J=J)

# The results are stored in response.states
print(response.states)

# If you want to see the result with subscript, use response.samples.
print([s for s in response.samples()])
print("------------------------------------------")
print("Numpy Example")
print("------------------------------------------")

## Numpy based interface
mat = np.array([[-1,-0.5,-0.5,-0.5],
                [-0.5,-1,-0.5,-0.5],
                [-0.5,-0.5,-1,-0.5],
                [-0.5,-0.5,-0.5,-1]])
print(mat)

bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype='SPIN')
print(bqm)

sampler = oj.SASampler()
response = sampler.sample(bqm)

print(response.states)
print(response.energies)
print("------------------------------------------")
print("Higher num_reads Example")
print("------------------------------------------")

# keys of h, J dictionaries can treat not only numbers.
h = {'a': -1, 'b': -1}
J = {('a', 'b'): -1, ('b', 'c'): 1}
# # Try solving 10 times by SA at a time. With the argument called num_reads.
sampler = oj.SASampler()
response = sampler.sample_ising(h, J, num_reads=10)
print(response.first.sample)
print(response.first.energy)
print(response.states)
print(response.energies)

print("------------------------------------------")
print("QUBO Example")
print("------------------------------------------")

Q = {(0, 0): -1, (0, 1): -1, (1, 2): 1, (2, 2): 1}
sampler = oj.SASampler()
response = sampler.sample_qubo(Q)
print(response.states)

print("------------------------------------------")
print("More Difficult QUBO Example")
print("------------------------------------------")

N = 100
Q = {(i, j): random.uniform(-1, 1) for i in range(N) for j in range(i+1, N)}

sampler = oj.SASampler()
response = sampler.sample_qubo(Q, num_reads=100)

plt.hist(response.energies, bins=15)
plt.xlabel('Energy', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()
print(response.first.sample)
print(response.first.energy)