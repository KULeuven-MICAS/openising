from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import ising.utils.adj as adj
import ising.generators as gen
from ising.model import IsingModel

# Generate data
seed=0
lbound, hbound = -1, 1
midbound = lbound + (hbound - lbound)/2
problem_size = 10
sim_time = 8

np.random.seed(seed)
initial_state = np.random.choice([lbound, hbound], size=problem_size).astype(np.int8)
model = gen.sample(adj.complete(problem_size), population=[-1, 0, 1], counts=[5, 1, 5], seed=seed)

def differential_coupling(state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    return -np.sum(coupling_matrix * (s - s.T), axis=0)

def binarisation(state, alpha=10):
    return np.tanh(alpha * np.tanh(alpha*state)) - state

def clamping(state, lbound=-1, hbound=1, alpha=100):
    return np.exp(alpha*(lbound - state)) - np.exp(alpha*(state - hbound))

def system_dynamics(t, state, coupling_matrix):
    return differential_coupling(state, coupling_matrix) \
        + 100 * (1 - np.exp(-0.05*t/sim_time)) * binarisation(state) \
        + 100 * clamping(state, lbound, hbound)

def evaluate_Hcont(state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    return - np.sum(coupling_matrix * (1 - 1/2 * (s - s.T)**2))

def evaluate_Hbin(state, coupling_matrix):
    lbound_diff = np.abs(state - lbound)
    hbound_diff = np.abs(state - hbound)
    rounded_state = np.where(lbound_diff < hbound_diff, -1, 1)
    return -np.dot(rounded_state.T, np.dot(coupling_matrix, rounded_state))

solution = solve_ivp(
        fun = system_dynamics,
        y0 = initial_state,
        t_span = (0, sim_time),
        args = (model.J,)
    )
times, states = solution.t, solution.y.T
continuous_energies = [evaluate_Hcont(state, model.J) for state in states]
binary_energies = [evaluate_Hbin(state, model.J) for state in states]

# Configure plots
plt.style.use('petroff10')
fig, (ax1, ax2) = plt.subplots(
                          nrows=2,
                          ncols=1,
                          sharex=True,
                          constrained_layout=True,
                          gridspec_kw={"height_ratios": [2, 1]}
                      )

ax1.plot(times, states)
ax1.axhline(lbound, color="black", lw=1)
ax1.axhline(midbound, color="black", ls="--", lw=1)
ax1.axhline(hbound, color="black", lw=1)
ax1.set_yticks([lbound, midbound, hbound], labels=[r"$x_{+}$", r"$x_{m}$", r"$x_{-}$"])
ax1.axvline(0, color="black", ls="--", lw=1)

ax2.plot(times, continuous_energies, label=r"$H_{cont}$")
ax2.plot(times, binary_energies, label=r"$H_{bin}$")
ax2.axvline(0, color="black", ls="--", lw=1)
ax2.set_yticks([])
ax2.set_xlabel(r"time")
ax2.legend()

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + ".png")
plt.savefig(SAVE_PATH, dpi=300)
plt.show()

