import time
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import ising.utils.adj as adj
import ising.generators as gen
from ising.model import IsingModel
from ising.solvers import ExhaustiveSolver
from ising.utils.numpy import triu_to_symm

seed=int(time.time()); print(seed)
problem_size = 128
end_time = 20

np.random.seed(seed)
initial_state = np.random.choice([-1, 1], size=problem_size).astype(float)
model = gen.sample(adj.complete(problem_size), population=[-1, 1], counts=[5, 5], seed=seed)
coupling_matrix = triu_to_symm(model.J)

def clamping(state):
    return np.where((state < -1) | (state > 1), 0, np.cos(np.pi / 2 * state)**2)

def clamping(state):
    clamping = np.zeros_like(state)
    clamping[(state < -1) | (state > 1)] = 0
    cond = (state >= -1) & (state < 0.75) | (state <=1) & (state > 0.75)
    clamping[cond] = np.cos(np.pi/2 * (4*state[cond]+3))**2
    clamping[(state >= -0.75) & (state <= 0.75)] = 1
    return clamping

def clamping(state, coupling):
    cond1 = (coupling > 0) & (state > 0)
    cond2 = (coupling < 0) & (state < 0)
    return np.where(cond1 | cond2, 1-state**2, 1)

# x = np.linspace(-1.1, 1.1, 10001)
# plt.plot(x, clamping(x, -1))
# plt.show()
# exit()

def restoration(state):
    restore = np.zeros_like(state)
    restore[state < -1] = -10 * (state[state < -1] - -1)
    restore[state > +1] = -10 * (state[state > +1] - +1)
    return restore

def coupling(state, coupling_matrix):
    k = np.tanh(3*state)
    return 1/2 * np.dot(coupling_matrix, k)

def system_dynamics(t, state):
    cu = coupling(state, coupling_matrix)
    return cu * clamping(state, cu)

def evaluate_Hbin(state, coupling_matrix):
    lbound_diff = np.abs(state + 1)
    hbound_diff = np.abs(state - 1)
    rounded_state = np.where(lbound_diff < hbound_diff, -1, 1)
    return model.evaluate(rounded_state)

solution = solve_ivp(
        fun = system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        t_eval = np.linspace(0, end_time, 1001)
    )
times, states = solution.t, solution.y.T
binary_energies = [evaluate_Hbin(state, model.J) for state in states]
best_energy = ExhaustiveSolver().solve(model)[1]


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

ax1.plot(times, states)
ax1.axhline(-1, color="black", lw=1)
ax1.axhline(0, color="black", ls="--", lw=1)
ax1.axhline(1, color="black", lw=1)
ax1.set_yticks([-1, 0, 1], labels=[r"$x_{+}$", r"$x_{m}$", r"$x_{-}$"])
ax1.axvline(0, color="black", ls="--", lw=1)

ax2.plot(times, binary_energies, label=r"$H_{bin}$", color=(255/255,169/255,14/255))
ax2.axhline(best_energy)
ax2.axvline(0, color="black", ls="--", lw=1)
ax2.set_xlabel(r"time")

plt.show()
