import numpy as np
import matplotlib.pyplot as plt

from ising.flow import TOP
from ising.benchmarks.parsers import G_parser
from ising.generators.MaxCut import MaxCut

def dvdt(t, v, coupling, Rsettle, Rflip, C):
    v[-1] = 1.0
    c = 1/np.abs(v)

    # ZIV diode
    z = v/Rsettle*(v-1)*(v+1) * (-63)

    # Flipping strength
    flip = np.block([1.-v[0], -1 - v[1], np.zeros((len(v)-2,))])

    dv = 1/C * (np.dot(coupling / Rsettle, c*v) - z + flip / Rflip)

    # Ensure the voltages stay in the range [-1, 1]
    cond1 = (dv > 0) & (v > 1)
    cond2 = (dv < 0) & (v < -1)
    dv *= np.where(cond1 | cond2, 0.0, 1.)
    dv[-1] = 0.
    return dv

def flipping(coupling, time_flip, dt, Rflip, Rsettle, C, state):
    time = 0.
    num_iter = int(time_flip/dt)
    all_states = np.zeros((num_iter+1, len(state)))
    all_states[0, :] = state
    for i in range(num_iter):
        k1 = dvdt(time , state, coupling, Rsettle, Rflip, C)
        k2 = dvdt(time + dt, state + dt*k1, coupling, Rsettle, Rflip, C)
        k3 = dvdt(time + dt/2, state + dt/4*(k1 + k2), coupling, Rsettle, Rflip, C)

        state += dt*(k1 / 6 + k2 / 6 + 2/3*k3)
        print(f"time {time:.2e}, dv[1]: {(k1[1] / 6 + k2[1] / 6 + 2/3*k3[1]):.4e}")
        time += dt
        all_states[i+1, :] = state
    return all_states

def test_worst_case():
    seed = 1
    np.random.seed(seed)
    # Create model with worse case scenario
    # I assume that the solver has 6 bit precision, max 8 bits
    N = 1000
    maxJ = 64
    J = np.ones((N, N))*maxJ
    np.fill_diagonal(J, 0)
    J[0,:] = -(maxJ - 1)*np.ones((N,))
    J = np.triu(J, k=1)
    J = J + J.T
    h = np.ones((N,1))*maxJ
    h[0, :] = -(maxJ-1)
    coupling = np.block([[J, h],[h.T, 0]])
    print(coupling)

    # Set the parameters for the Multiplicative solver
    Kmin = 1
    Capacitance = 1
    tau_settle = Kmin*Capacitance/(maxJ*N)
    tau_flip = tau_settle/10
    Rflip = tau_flip / Capacitance
    print(f"tau_settle: {tau_settle:.2e}, tau_flip: {tau_flip:.2e}, Rflip: {Rflip:.2e}")

    dt = 1e-7
    time_flip = 2e-4
    time_points = np.arange(0, time_flip+dt, dt)
    state = np.ones((N+1,))
    state[0] = -1.0
    all_states = flipping(coupling, time_flip, dt, Rflip, Kmin, Capacitance, state)

    plt.figure()
    plt.plot(time_points, all_states[:, 1:])
    plt.plot(time_points, all_states[:, 0], "--b")
    plt.xlabel("Time (s)")
    plt.ylabel("State (V)")
    plt.savefig("./flipping.png")
    plt.close()

def test_any_case():
    graph, best_found = G_parser(TOP / "ising/benchmarks/G/K2000.txt")
    model = MaxCut(graph)
    coupling = model.J + model.J.T
    h = model.h.reshape((model.num_variables, 1))
    coupling = np.block([[coupling, h],[h.T, 0]])

    Rsettle = 1
    Capacitance = 1
    Rflip = Rsettle / (1e2*128)
    max_Jav = np.max(np.sum(np.abs(coupling), axis=1))
    tau_settle = Rsettle * Capacitance / max_Jav
    tau_flip = Rflip * Capacitance

    print(f"tau_settle: {tau_settle:.2e}, tau_flip: {tau_flip:.2e}, Rflip: {Rflip:.2e}")

    dt = 1e-8
    time_flip = 7e-4
    state = np.random.uniform(-1, 1, (model.num_variables+1,))
    state[-1] = 1.0
    state[0] = -1.0
    state[1] = 1.01
    all_states = flipping(coupling, time_flip, dt, Rflip, Rsettle, Capacitance, state)

    plt.figure()
    plt.plot(np.arange(0, time_flip + 0.1*dt, dt), all_states[:, 2:])
    plt.plot(np.arange(0, time_flip + 0.1*dt, dt), all_states[:, 0], "--b")
    plt.plot(np.arange(0, time_flip + 0.1*dt, dt), all_states[:, 1], "--g")
    plt.xlabel("Time (s)")
    plt.ylabel("State (V)")
    plt.savefig("./flipping_any_case.png")
    plt.close()



if __name__ == "__main__":
    # test_worst_case()
    test_any_case()

