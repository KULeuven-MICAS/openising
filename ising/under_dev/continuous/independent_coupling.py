import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import ising.utils.adj as adj
import ising.generators as gen
from ising.model import IsingModel

def diff_coupling(coupling_matrix, state):
    s = np.array([state] * state.shape[0])
    return -np.sum(coupling_matrix * (s - s.T), 0)

def product_coupling(coupling_matrix, state):
    return np.dot(coupling_matrix, state)

def simulate_system(coupling_dynamics, coupling_matrix, initial_state, t_end=10, t_points=100, oob_penalty=1, lbound=0, hbound=1):

    def system_dynamics(t, state):
       return coupling_dynamics(coupling_matrix, state) + oob_penalty * np.exp(100 * (lbound - state)) - np.exp(100*(state - hbound))

    solution = solve_ivp(
            fun = system_dynamics,
            t_span = (0, t_end),
            y0 = initial_state,
            t_eval = np.linspace(0, t_end, t_points)
        )

    return solution.t, solution.y

def round_to_nearest(state, lbound=-1, hbound=1):
    lbound_diff = np.abs(state - lbound)
    hbound_diff = np.abs(state - hbound)
    return np.where(lbound_diff < hbound_diff, lbound, hbound)

def binary_H(state, coupling_matrix, lbound=-1, hbound=1):
    state = round_to_nearest(state, lbound, hbound)
    return -np.dot(state.T, np.dot(coupling_matrix, state))

def p_cont_H(state, coupling_matrix):
    return -np.dot(state.T, np.dot(coupling_matrix, state))

def d_cont_H(state, coupling_matrix):
    s = np.array([state] * state.shape[0])
    return -np.sum(coupling_matrix * (1 - 1/2 * (s - s.T)**2))

if __name__ == "__main__":
    seed = 0
    problem_size = 14
    lbound, hbound = -1, 1
    t_end=10
    t_points=3000
    oob_penalty=100

    #np.random.seed(seed)
    initial_state = np.random.choice([lbound, hbound], size=problem_size).astype(np.int8)
    model = gen.sample(adj.complete(problem_size), population=[-2, -1, 0, 1, 2], counts=[2, 5, 1, 5, 2], seed=seed)

    p_times, p_states = simulate_system(product_coupling, model.J, initial_state, t_end, t_points, oob_penalty, lbound, hbound)
    print(f"Solution found by product_coupling is {round_to_nearest(p_states.T[-1]).astype(int)}")

    d_times, d_states = simulate_system(diff_coupling, model.J, initial_state, t_end, t_points, oob_penalty, lbound, hbound)
    print(f"Solution found by diff_coupling is    {round_to_nearest(d_states.T[-1]).astype(int)}")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.05, wspace=0.03)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p_times, p_states.T)
    ax1.axhline(lbound + (hbound-lbound)/2, ls=':')
    ax1.set_ylabel(r"state")
    ax1.set_title("product coupling")
    ax1.tick_params(labelbottom=False, bottom=False, which="both")

    continuous_energies = [p_cont_H(state, model.J) for state in p_states.T]
    binary_energies = [binary_H(state, model.J) for state in p_states.T]
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.sharex(ax1)
    ax2.plot(p_times, continuous_energies, label=r"$H_{cont}$")
    ax2.plot(p_times, binary_energies, label=r"$H_{bin}$")
    ax2.set_ylabel(r"energy")
    ax2.set_xlabel(r"time")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.sharey(ax1)
    ax3.plot(d_times, d_states.T)
    ax3.axhline(lbound + (hbound-lbound)/2, ls=':')
    ax3.set_title("differential coupling")
    ax3.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False, which="both")

    continuous_energies = [d_cont_H(state, model.J) for state in d_states.T]
    binary_energies = [binary_H(state, model.J) for state in d_states.T]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.sharex(ax3)
    ax4.sharey(ax2)
    ax4.plot(d_times, continuous_energies, label=r"$H_{cont}$")
    ax4.plot(d_times, binary_energies, label=r"$H_{bin}$")
    ax4.set_xlabel(r"time")
    ax4.tick_params(labelleft=False, left=False, which="both")
    ax4.legend()

    plt.show()
