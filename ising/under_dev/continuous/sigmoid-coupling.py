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


# Generate data
# def system_dynamics(t, state, coupling_matrix, coupling_f):
#     lower_clamping = (0.5 + np.tanh(50*(-1-state))/2) * np.exp(1*(-1 - state))
#     upper_clamping = - (0.5 + np.tanh(50*(state-1))/2) * np.exp(1*(state - 1))
#     coupling = np.dot(coupling_matrix, coupling_f(state))
#     return coupling + 100 * (lower_clamping + upper_clamping)

def clamping(state, coupling):
    cond1 = (coupling > 0) & (state > 0)
    cond2 = (coupling < 0) & (state < 0)
    return np.where(cond1 | cond2, 1-state**2, 1)

def system_dynamics(t, state, coupling_matrix, coupling_f):
    coupling = np.dot(coupling_matrix, coupling_f(state))
    return coupling * clamping(state, coupling)

def evaluate_Hbin(state, coupling_matrix):
    lbound_diff = np.abs(state - lbound)
    hbound_diff = np.abs(state - hbound)
    rounded_state = np.where(lbound_diff < hbound_diff, -1, 1)
    return -1/2 * np.dot(rounded_state.T, np.dot(coupling_matrix, rounded_state))

def evaluate_Hcont(state, coupling_matrix):
    return -1/2 * np.dot(state.T, np.dot(coupling_matrix, state))

def identity(x):
    return x

def sigmoid(x, k=3):
    return np.tanh(k*x)

def differential_system_dynamics(t, state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    coupling = -np.sum(coupling_matrix * (s - s.T), axis=0)
    return coupling * clamping(state, coupling)

seed=int(time.time())
print(f"seed: {seed}")
lbound, hbound = -1, 1
midbound = lbound + (hbound - lbound)/2
problem_size = 20
end_time = 3
sigmoid_k = 20

np.random.seed(seed)
initial_state = np.random.choice([lbound, hbound], size=problem_size).astype(np.int8)
model = gen.sample(adj.complete(problem_size), population=[-1, 0, 1], counts=[5, 1, 5], seed=seed)
coupling_matrix = triu_to_symm(model.J)

solution = solve_ivp(
        fun = differential_system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        args = (coupling_matrix,),
        dense_output=True
    )
diff_times, diff_states = solution.t, solution.y.T
diff_binary_energies = [evaluate_Hbin(state, coupling_matrix) for state in diff_states]
print("finished differential simulation")

solution = solve_ivp(
        fun = system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        t_eval = np.linspace(0, end_time, 1001),
        args = (coupling_matrix, identity),
        dense_output=True
    )
times, states = solution.t, solution.y.T
continuous_energies = [evaluate_Hcont(state, coupling_matrix) for state in states]
binary_energies = [evaluate_Hbin(state, coupling_matrix) for state in states]
print("finished multiplicative linear simulation")

solution = solve_ivp(
        fun = system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        t_eval = np.linspace(0, end_time, 1001),
        args = (coupling_matrix, partial(sigmoid, sigmoid_k)),
        dense_output=True
    )
times2, states2 = solution.t, solution.y.T
continuous_energies2 = [evaluate_Hcont(state, coupling_matrix) for state in states2]
binary_energies2 = [evaluate_Hbin(state, coupling_matrix) for state in states2]
print("finished multiplicative sigmoid simulation")

best_energy = ExhaustiveSolver().solve(model)[1]
print("finished exhaustive solve")


# Configure plots
plt.style.use('petroff10')
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
                          nrows=2,
                          ncols=3,
                          sharex=True,
                          constrained_layout=True,
                          gridspec_kw={"height_ratios": [2, 1]},
                          figsize=(18, 6)
                      )

ax1.plot(diff_times, diff_states)
ax1.axhline(lbound, color="black", lw=1)
ax1.axhline(midbound, color="black", ls="--", lw=1)
ax1.axhline(hbound, color="black", lw=1)
ax1.set_yticks([lbound, midbound, hbound], labels=[r"$x_{+}$", r"$x_{m}$", r"$x_{-}$"])
ax1.axvline(0, color="black", ls="--", lw=1)
ax1.set_title(r"Differential coupling (BRIM)")

ax4.plot(diff_times, diff_binary_energies, label=r"$H_{bin}$", color=(255/255,169/255,14/255))
ax4.axhline(best_energy, label="best solution", color=(189/255, 31/255, 1/255))
ax4.axvline(0, color="black", ls="--", lw=1)
ax4.set_xlabel(r"time")
ax4.legend()

ax2.plot(times, states)
ax2.axhline(lbound, color="black", lw=1)
ax2.axhline(midbound, color="black", ls="--", lw=1)
ax2.axhline(hbound, color="black", lw=1)
ax2.set_yticks([])
ax2.axvline(0, color="black", ls="--", lw=1)
ax2.set_title(r"Linear coupling function")

ax5.sharey(ax4)
ax5.plot(times, continuous_energies, label=r"$H_{cont}$")
ax5.plot(times, binary_energies, label=r"$H_{bin}$")
ax5.axhline(best_energy, label="best solution", color=(189/255, 31/255, 1/255))
ax5.axvline(0, color="black", ls="--", lw=1)
ax5.set_xlabel(r"time")
ax5.legend()

ax3.plot(times2, states2)
ax3.axhline(lbound, color="black", lw=1)
ax3.axhline(midbound, color="black", ls="--", lw=1)
ax3.axhline(hbound, color="black", lw=1)
ax3.set_yticks([])
ax3.set_xticks([])
ax3.axvline(0, color="black", ls="--", lw=1)
ax3.set_title(f"Sigmoid with k={sigmoid_k}")

ax6.sharey(ax4)
ax6.plot(times2, continuous_energies2, label=r"$H_{cont}$")
ax6.plot(times2, binary_energies2, label=r"$H_{bin}$")
ax6.axhline(best_energy, label="best solution", color=(189/255, 31/255, 1/255))
ax6.axvline(0, color="black", ls="--", lw=1)
ax6.set_xlabel(r"time")
ax6.legend()

# Save and plot
plt.show()
