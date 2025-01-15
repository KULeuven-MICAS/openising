import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import ising.utils.adj as adj
import ising.generators as gen
from ising.model import IsingModel


def diff_coupling(coupling_matrix, state):
    s = np.array([state] * state.shape[0])
    return -np.sum(coupling_matrix * (s - s.T), 0)


if __name__ == "__main__":
    seed = 0
    problem_size = 10
    lbound, hbound = -1, 1
    t_end, t_points = 10, 3000
    oob_penalty=100

    initial_state = np.random.choice([lbound, hbound], size=problem_size).astype(np.int8)
    coupling = gen.sample(adj.complete(problem_size), population=[-1, 0, 1], counts=[5, 1, 5]).J

    def system_dynamics(t, state):
       return diff_coupling(coupling, state) + oob_penalty * np.exp(10 * (lbound - state)) - np.exp(10*(state - hbound))

    solution = solve_ivp(
            fun = system_dynamics,
            t_span = (0, t_end),
            y0 = initial_state,
            t_eval = np.linspace(0, t_end, t_points)
        )
    times, states = solution.t, solution.y

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax1.plot(times, states.T)
    ax1.axhline(0, ls=':')

#plt.plot(x, y, lw=2)
#
## Add labels, title, and legend
#plt.xlabel(r"t")
#plt.ylabel(r"$x_i$")
#plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x-axis
#plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # y-axis
#
## Show the plot
#plt.tight_layout()
#plt.show()
##plt.savefig('/mnt/c/Users/tbettens/Pictures/plt/diff-coupling.png')
