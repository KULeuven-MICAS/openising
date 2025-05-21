import numpy as np
import matplotlib.pyplot as plt
import pathlib

from ising.generators.Knapsack import knapsack
from ising.solvers.Multiplicative import Multiplicative
from ising.postprocessing.energy_plot import plot_energies
from ising.postprocessing.plot_solutions import plot_state
from ising.solvers.SB import ballisticSB
from ising.solvers.exhaustive import ExhaustiveSolver

np.random.seed(1)

def compute_tau(y, nb_bits):
    y_bit = bin(y)[2:]
    tau = np.array([1 if y_bit[bit] == "1" else -1 for bit in range(len(y_bit)-1, -1, -1)])
    if len(tau) < nb_bits:
        tau = np.concatenate((tau, -np.ones((nb_bits - len(tau)),)))
    return tau

def original_energy(profit, capacity, weights, x, y, penalty_val):
    N = len(weights)
    alpha = np.max(profit)*penalty_val
    energy = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                energy -= profit[i,j]/2 * x[i] * x[j]
            else:
                energy -= profit[i,j] * x[i] * x[j]

    constraint = 0
    for i in range(N):
        constraint += weights[i]*x[i]
    print(f"profit: {energy}")
    print(f"constraint: {constraint + y - capacity}")

    energy = energy + alpha * (constraint + y - capacity)**2 #+ (y**2)*penalty_val
    return energy

def original_energy_ising(profit, capacity, weights, sigma, penalty_val):
    N = len(weights)
    nb_bits = int(np.floor(np.log2(capacity) + 1))
    alpha = penalty_val * np.max(profit)
    energy = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                energy -= profit[i,j]*(sigma[i] + 1)/2*(sigma[j] + 1)/2
    for i in range(N):
        energy -= profit[i,i]*(sigma[i] + 1)/2
    
    constraint = 0
    for i in range(N):
        constraint += weights[i]*(sigma[i] + 1)/2
    for q in range(nb_bits):
        constraint += (2**(q))*(sigma[N+q] + 1)/2
    constraint -= capacity

    y_constraint = 0
    # for q in range(nb_bits):
    #     y_constraint += (2**(q))*(sigma[N+q] + 1)/2
    energy += alpha * (constraint**2) + (y_constraint**2)*penalty_value
    
    return energy

capacity = 15
nb_bits = int(np.floor(np.log2(capacity) + 1))

N = 5
profit = np.array([[4, 3, 8, 0, 14],
                   [3, 2, 0, 12, 0],
                   [8, 0, 2, 0, 9],
                   [0, 12, 0, 10, 5],
                   [14, 0, 9, 5, 1]])

weights = np.array([12, 2, 1, 4, 1])
penalty_value = 1

model_ising = knapsack(profit, capacity, weights, penalty_value)
print(np.count_nonzero(model_ising.J + model_ising.J.T)/(model_ising.num_variables**2))
print(np.max(model_ising.h))
print(np.min(model_ising.J))
print(np.min(model_ising.h))
print("=============== Exhaustive solution ================")

state_optim, energy_optim = ExhaustiveSolver().solve(model_ising)
print(f"state: {state_optim} and energy: {energy_optim}")
y = np.sum([2**q*(state_optim[N+q]+1)/2 for q in range(nb_bits)])
# print(f"Optimal energy: {energy_optim} and state: {state_optim}")
print(original_energy(profit, capacity, weights, np.where(state_optim[:N] == -1, 0, 1), y, penalty_value))

print("=============== Actual optimum ================")
print(original_energy(profit, capacity, weights, np.array([0, 1, 1, 1, 1]), 7, penalty_value))

state, energy = ballisticSB().solve(model_ising, np.zeros((N+nb_bits,)), 10000, 10, 0.0001, 1.0, "./bSB_knapsack_test.log")
plot_state("bSB", "./bSB_knapsack_test.log", "bSB_state", pathlib.Path("./"))
plot_energies(pathlib.Path("./bSB_knapsack_test.log"), "bSB_energy_knapsack", best_found=None)

N = model_ising.num_variables
num_states = 2**N

# Generate all possible states
states = []
energies = []
optimal_index = -1
received_index = -1
received_energy = energy
# Use binary numbers to generate all combinations of -1 and 1
for i in range(num_states):
    # Convert number to binary and pad with zeros
    binary = format(i, f'0{N}b')
    # Convert 0s and 1s to -1s and 1s
    current_state = np.array([1 if int(b) else -1 for b in binary])
    current_energy = model_ising.evaluate(current_state)
    if current_energy == energy_optim:
        optimal_index = i
    if np.all(current_state == state):
        received_index = i
        received_energy = current_energy
    if current_energy <= energy_optim:
        print(f"Found state with lower energy: {current_energy} at state {current_state}")
    
    states.append(current_state)
    energies.append(current_energy)

# Convert to numpy arrays for easier handling
states = np.array(states)
energies = np.array(energies)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_states), energies, 'b-', label='Energy')
plt.axvline(optimal_index, ymin=min(energies)-100, ymax=max(energies)+100, color='r', linestyle='-', label=f'Optimal State: {energy_optim:.2f}')
plt.axvline(received_index, ymin=min(energies)-100, ymax=max(energies)+100, color='y', linestyle='--', label=f'Received State: {received_energy}')
plt.axhline(y=energies[optimal_index], color='g', linestyle='--', label='Energy of Optimal State')
plt.xlabel('State Index')
plt.ylabel('Energy')
plt.title(f'Energy Landscape for {N} Variables')
plt.grid(True)
plt.legend()
plt.savefig("./energy_landscape.png")
plt.close()