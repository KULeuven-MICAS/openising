from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import ising.utils.adj as adj
import ising.generators as gen
from ising.model import IsingModel


# Generate data
seed = 0
lbound, hbound = -1, 1
midbound = lbound + (hbound - lbound)/2
problem_size = 16
end_time = 8

np.random.seed(seed)
initial_state = np.random.choice([lbound, hbound], size=problem_size).astype(np.int8)
model = gen.sample(adj.complete(problem_size), population=[-1, 0, 1], counts=[5, 1, 5], seed=seed)

def differential_coupling(state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    return -np.sum(coupling_matrix * (s - s.T), axis=0)

def product_coupling(coupling_matrix, state):
    return np.dot(coupling_matrix, state)

def clamping(state, lbound=-1, hbound=1, alpha=10):
    #return np.exp(alpha*(lbound - state)) - np.exp(alpha*(state - hbound))
    lower_clamping = (0.5 + np.tanh(100*(lbound-state))/2) * np.exp(alpha*(lbound - state))
    upper_clamping = - (0.5 + np.tanh(100*(state-hbound))/2) * np.exp(alpha*(state - hbound))
    return lower_clamping + upper_clamping

def evaluate_Hbin(state, coupling_matrix):
    lbound_diff = np.abs(state - lbound)
    hbound_diff = np.abs(state - hbound)
    rounded_state = np.where(lbound_diff < hbound_diff, -1, 1)
    return -np.dot(rounded_state.T, np.dot(coupling_matrix, rounded_state))

def evaluate_Hcont_differential(state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    return - np.sum(coupling_matrix * (1 - 1/2 * (s - s.T)**2))

def evaluate_Hcont_product(state, coupling_matrix):
    return -np.dot(state.T, np.dot(coupling_matrix, state))

def differential_system_dynamics(t, state, coupling_matrix):
    return differential_coupling(state, coupling_matrix) + 10 * clamping(state, lbound, hbound)

solution = solve_ivp(
        fun = differential_system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        args = (model.J,)
    )
diff_times, diff_states = solution.t, solution.y.T
diff_continuous_energies = [evaluate_Hcont_differential(state, model.J) for state in diff_states]
diff_binary_energies = [evaluate_Hbin(state, model.J) for state in diff_states]

def product_system_dynamics(t, state, coupling_matrix):
    return product_coupling(coupling_matrix, state) + 10 * clamping(state, lbound, hbound)

solution = solve_ivp(
        fun = product_system_dynamics,
        y0 = initial_state,
        t_span = (0, end_time),
        args = (model.J,)
    )
prod_times, prod_states = solution.t, solution.y.T
prod_continuous_energies = [evaluate_Hcont_product(state, model.J) for state in prod_states]
prod_binary_energies = [evaluate_Hbin(state, model.J) for state in prod_states]


# Configure plots
plt.style.use('petroff10')
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
                          nrows=2,
                          ncols=2,
                          sharex=True,
                          constrained_layout=True,
                          gridspec_kw={"height_ratios": [2, 1]},
                          figsize=(12, 6)
                      )

ax1.plot(diff_times, diff_states)
ax1.axhline(lbound, color="black", lw=1)
ax1.axhline(midbound, color="black", ls="--", lw=1)
ax1.axhline(hbound, color="black", lw=1)
ax1.set_yticks([lbound, midbound, hbound], labels=[r"$x_{+}$", r"$x_{m}$", r"$x_{-}$"])
ax1.axvline(0, color="black", ls="--", lw=1)
ax1.set_title(r"Differential coupling", fontsize=16)
ax1.set_ylabel(r"states", rotation=0)

ax2.sharey(ax4)
ax2.plot(diff_times, diff_continuous_energies, label=r"$H_{cont}$")
ax2.plot(diff_times, diff_binary_energies, label=r"$H_{bin}$")
ax2.axvline(0, color="black", ls="--", lw=1)
ax2.set_xlabel(r"time")
ax2.set_yticks([])
ax2.legend()

ax3.plot(prod_times, prod_states)
ax3.axhline(lbound, color="black", lw=1)
ax3.axhline(midbound, color="black", ls="--", lw=1)
ax3.axhline(hbound, color="black", lw=1)
ax3.set_yticks([])
ax3.set_xticks([])
ax3.axvline(0, color="black", ls="--", lw=1)
ax3.set_title(r"Multiplicative coupling", fontsize=16)

ax4.plot(prod_times, prod_continuous_energies, label=r"$H_{cont}$")
ax4.plot(prod_times, prod_binary_energies, label=r"$H_{bin}$")
ax4.axvline(0, color="black", ls="--", lw=1)
ax4.set_xlabel(r"time")
ax4.set_yticks([])
ax4.legend()

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + str(seed) + ".png")
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.show()
