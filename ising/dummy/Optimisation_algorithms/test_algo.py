from SCA import SCA
from SA import SA
from SB import ballisticSB, discreteSB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import openjij as oj
import helper_functions as hf
import math
import BLIM
from scipy.integrate import solve_ivp
# from  ising.model import BinaryQuadraticModel

VERBOSE = False
VERBOSE_SOLVER = False
VERBOSE_PLOT = False


def run_solver(
    solver: str,
    s_init: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    S: int,
    N: int,
    dir: str = ".",
    G=None,
    plt:bool=False,
    **hyperparameters,
):
    """
    Runs the solver and plots the energy during optimisation

    :param str solver: solver name (SCA, SA, bSB or dSB)
    :param np.ndarray s_init: initial spin configuration
    :param np.ndarray J: interaction coefficients
    :param np.ndarray h: magnetic field coefficients
    :param int S: total amount of iterations
    :param int N: size of the problem
    :param str dir: directory where the figures should be stored
    :param hyperparameters: stores hyperparameters of specific solver
    :return sigma_optim (np.ndarray): optimal spin configuration
    :return energies (list): energies during optimisation
    """
    if solver == "SCA":
        sigma_optim, energies = SCA(
            s_init=s_init,
            J=J,
            h_init=h,
            S=S,
            q=hyperparameters["q"],
            T=hyperparameters["T"],
            r_q=hyperparameters["r_q"],
            r_t=hyperparameters["r_t"],
            verbose=VERBOSE_SOLVER,
        )
    elif solver[1:] == "SB":
        x_init = np.multiply(
            np.random.uniform(low=0.0, high=0.5, size=np.shape(s_init)), s_init
        )
        y_init = np.random.uniform(-0.5, 0.5, size=np.shape(s_init))
        if solver[0] == "b":
            sigma_optim, energies, times = ballisticSB(
                h=h,
                J=J,
                x_init=x_init,
                y_init=y_init,
                dt=hyperparameters["dt"],
                Nstep=S,
                a0=hyperparameters["a0"],
                c0=hyperparameters["c0"],
                at=hyperparameters["at"],
                verbose=VERBOSE_SOLVER,
            )
        else:
            sigma_optim, energies, times = discreteSB(
                h=h,
                J=J,
                x_init=x_init,
                y_init=y_init,
                dt=hyperparameters["dt"],
                Nstep=S,
                a0=hyperparameters["a0"],
                c0=hyperparameters["c0"],
                at=hyperparameters["at"],
                verbose=VERBOSE_SOLVER,
            )

    else:
        sigma_optim, energies = SA(
            T=hyperparameters["T"],
            r_T=hyperparameters["r_t"],
            S=S,
            J=J,
            h=h,
            sigma=s_init,
            verbose=VERBOSE_SOLVER,
        )
    energy = hf.compute_energy(J, h, sigma_optim) + 1/2*np.sum(J)
    print(f"The optimal energy of {solver} is: {energy}")
    if N <= 20:
        print(f"The optimal state of {solver}: {sigma_optim}")
        hf.plot_solution(sigma_optim, G, solver)
    if plt:
        hf.plot_energies({solver: energies}, S, f"{dir}\Energy_{solver}.png")
    return sigma_optim, energies


def problem1():
    print("--------------------------------------")
    print("Not-fully connected graph with 8 nodes")
    print("--------------------------------------")
    data = np.array(
        [
            [0, 1, 1.0],
            [0, 3, 1.0],
            [1, 2, 1.0],
            [2, 3, 1.0],
            [2, 7, 1.0],
            [3, 6, 1.0],
            [4, 5, 1.0],
            [4, 6, 1.0],
            [4, 7, 1.0],
            [5, 6, 1.0],
            [6, 7, 1.0],
        ]
    )

    N = 8
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    for row in data:
        i, j = row[0], row[1]
        G.add_edge(i, j)
    nx.draw_networkx(G)
    folder = "..\output"

    print("Setting hyperparameters")
    J, h = hf.get_coeffs_from_array_MC(N, data)
    nb_runs = 10
    S = 500
    T = 50.0
    q = 1.0
    T_end = 0.05
    q_end = 3.0
    r_t = hf.compute_rx(T, T_end, S)
    r_q = hf.compute_rx(q, q_end, S)
    dt = 0.25
    a0 = 1.0
    c0 = 0.5 / (math.sqrt(N) * math.sqrt(np.sum(np.power(J, 2)) / (N * (N - 1))))

    def at(t):
        return a0 / (S * dt) * t

    s_init = hf.get_random_s(N)
    print(f"Initial sigma: {s_init}")

    mat = np.diag(h) - J
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype="SPIN")
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=nb_runs)
    hf.plot_energy_dist(
        response.energies, "SA OpenJij", folder + "\histogram_sa_openjij.png"
    )

    print(f"Solution of OpenJij: {response.first.sample}")
    print(f"Optimal energy of OpenJij:  {response.first.energy}")
    hf.plot_solution(response.first.sample, G, "SA OpenJij")

    energy_sca = []
    energy_sa = []
    energy_bsb = []
    energy_dsb = []
    for i in range(nb_runs):
        sigma_sca, energies_sca = run_solver(
            solver="SCA",
            s_init=s_init,
            J=J,
            h=h,
            S=S,
            N=N,
            G=G,
            dir=folder,
            q=q,
            r_q=r_q,
            T=T,
            r_t=r_t,
        )
        energy_sca.append(energies_sca[-1])
        sigma_SA, energies_sa = run_solver(
            solver="SA",
            s_init=s_init,
            J=J,
            h=h,
            S=S,
            N=N,
            G=G,
            dir=folder,
            T=T,
            r_t=r_t,
        )
        energy_sa.append(energies_sa[-1])
        sigma_bsb, energies_bsb = run_solver(
            solver="bSB",
            s_init=s_init,
            J=J,
            h=h,
            S=S,
            N=N,
            G=G,
            dir=folder,
            a0=a0,
            c0=c0,
            at=at,
            dt=dt,
        )
        energy_bsb.append(energies_bsb[-1])
        sigma_dsb, energies_dsb = run_solver(
            solver="dSB",
            s_init=s_init,
            J=J,
            h=h,
            S=S,
            N=N,
            G=G,
            dir=folder,
            a0=a0,
            c0=c0,
            at=at,
            dt=dt,
        )
        energy_dsb.append(energies_dsb[-1])

    hf.plot_energy_dist(energy_sca, "SCA", folder + "\histogram_sca.png")
    hf.plot_energy_dist(energy_sa, "SA", folder + "\histogram_sa.png")
    hf.plot_energy_dist(energy_bsb, "bSB", folder + "\histogram_bSB.png")
    hf.plot_energy_dist(energies_dsb, "dSB", folder + "\histogram_dSB.png")

    energies = {
        "SCA": energies_sca,
        "SA": energies_sa,
        "bSB": energies_bsb,
        "dSB": energies_dsb,
    }
    hf.plot_energies(energies, S, folder + "\energies_all.png")


def G1():
    print("--------------------------------------")
    print("Not-fully connected graph from G1 benchmark")
    print("--------------------------------------")
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    file = os.path.join(parent_dir, "G1.txt")
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0, 0])
    nb_edges = int(info[0, 1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = hf.get_coeffs_from_array_MC(N, info)
    s_init = hf.get_random_s(N)

    folder = "../output/algo_problem2"
    S = 10000
    q = 1.0
    T = 50.0
    T_end = 0.05
    q_end = 5.0
    r_q = hf.compute_rx(q, q_end, S)
    r_t = hf.compute_rx(T, T_end, S)
    tend = 1000
    dt = tend / S
    a0 = 1
    c0 = 0.5 / (math.sqrt(N) * math.sqrt(np.sum(np.power(J, 2)) / (N * (N - 1))))

    def at(t):
        return a0 / (S * dt) * t

    sigma_sca, energies_sca = run_solver(
        solver="SCA",
        s_init=s_init,
        J=J,
        h=h,
        S=S,
        T=T,
        r_t=r_t,
        N=N,
        q=q,
        r_q=r_q,
        dir=folder,
    )
    sigma_sca = np.copy(sigma_sca)
    _, energies_bSB = run_solver(
        solver="bSB",
        s_init=s_init,
        J=J,
        h=h,
        S=S,
        N=N,
        dir=folder,
        a0=a0,
        c0=c0,
        at=at,
        dt=dt,
    )
    _, energies_dSB = run_solver(
        solver="dSB",
        s_init=s_init,
        J=J,
        h=h,
        S=S,
        N=N,
        dir=folder,
        a0=a0,
        c0=c0,
        at=at,
        dt=dt,
    )

    mat = np.diag(h) - J
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype="SPIN")
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=500)
    print(response.first.energy)

    hf.plot_energies(
        {"SCA": energies_sca, "bSB": energies_bSB, "dSB": energies_dSB},
        S=S,
        filename=folder + "\energies_all.png",
    )


def k2000():
    print("--------------------------------------")
    print("Test K2000 benchmark")
    print("--------------------------------------")

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    file = os.path.join(parent_dir, "WK2000_1.txt")
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0, 0])
    nb_edges = int(info[0, 1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = hf.get_coeffs_from_array_MC(N, info)
    s_init = hf.get_random_s(N)

    print("Hyperparameter setup")
    S = 1000
    T = 50.0
    T_end = 0.05
    r_t = hf.compute_rx(T, T_end, S)
    q = 2.0
    q_end = 7.0
    r_q = hf.compute_rx(q, q_end, S)
    c0 = 0.5 / (math.sqrt(N) * math.sqrt(np.sum(np.power(J, 2)) / (N * (N - 1))))
    a0 = 1
    dt = 0.25

    def at(t):
        return a0 / (S * dt) * t

    print("SCA solver")
    _, energies_SCA = run_solver(
        solver="SCA",
        s_init=s_init,
        J=J/2,
        h=h,
        S=S,
        N=N,
        dir=parent_dir + "\output\K2000_test",
        T=T,
        r_t=r_t,
        q=q,
        r_q=r_q,
    )
    print("bSB solver")
    _, energies_bSB = run_solver(
        solver="bSB",
        s_init=s_init,
        J=J,
        h=h,
        S=S,
        N=N,
        dir=parent_dir + "\output\K2000_test",
        a0=a0,
        c0=c0,
        dt=dt,
        at=at,
    )
    print("dSB solver")
    _, energies_dSB = run_solver(
        solver="dSB",
        s_init=s_init,
        J=J,
        h=h,
        S=S,
        N=N,
        dir=parent_dir + "\output\K2000_test",
        a0=a0,
        c0=c0,
        dt=dt,
        at=at,
    )

    print("SA OpenJij solver")
    mat = np.diag(h) - J/2
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype="SPIN")
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=10)
    print("OpenJij best energy: " + str(response.first.energy))

    hf.plot_energies(
        energies={
            "SCA": energies_SCA,
            "bSB": energies_bSB,
            "dSB": energies_dSB,
            'SA OpenJij': response.first.energy
        },
        x=list(range(S)),
        xname='iteration',
        filename=parent_dir + "\output\K2000_test\energies_all.png",
    )

def test_SB():
    print("--------------------------------------")
    print("Test SB implementation")
    print("--------------------------------------")

    print('load K2000 max-cut benchmark')
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    file = os.path.join(parent_dir, "WK2000_1.txt")
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0, 0])
    nb_edges = int(info[0, 1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = hf.get_coeffs_from_array_MC(N, info)
    s_init = hf.get_random_s(N)

    Nstep_list = [500, 1000, 5000]
    dt = 1
    a0 = 1
    c0 = 0.5 / (math.sqrt(N) * math.sqrt(np.sum(np.power(J, 2)) / (N * (N - 1))))
    def at(t):
        return a0 / (dt * Nstep) * t
    folder = parent_dir + '\output\SB_test'

    nb_runs = 20
    energies_bSB = {500: [], 1000: [], 5000: []}
    averages_bSB = []
    min_bSB = []
    max_bSB = []
    energies_dSB = {500: [], 1000: [], 5000: []}
    averages_dSB = []
    min_dSB = []
    max_dSB = []
    for Nstep in Nstep_list:
        for i in range(nb_runs):
            print('run: ' + str(i))
            _, energy_bSB = run_solver(solver='bSB', s_init=s_init, J=J, h=h, S=Nstep, N=N, dir=folder, dt=dt, a0=a0, c0=c0, at=at)
            _, energy_dSB = run_solver(solver='dSB', s_init=s_init, J=J, h=h, S=Nstep, N=N, dir=folder, dt=dt, a0=a0, c0=c0, at=at)
            energies_bSB[Nstep].append(energy_bSB[-1] + 1/2*np.sum(J))
            energies_dSB[Nstep].append(energy_dSB[-1] + 1/2*np.sum(J))
        averages_bSB.append(np.mean(energies_bSB[Nstep]))
        averages_dSB.append(np.mean(energies_dSB[Nstep]))
        min_bSB.append(np.min(energies_bSB[Nstep]))
        max_bSB.append(np.max(energies_bSB[Nstep]))
        min_dSB.append(np.min(energies_dSB[Nstep]))
        max_dSB.append(np.max(energies_dSB[Nstep]))

    bestknown = -33337
    plt.figure()
    plt.plot(Nstep_list, averages_bSB, 'b--', label='bSB')
    plt.plot(Nstep_list, averages_dSB, 'r--', label='dSB')
    plt.fill_between(Nstep_list, min_bSB, max_bSB, color='blue', alpha=.1)
    plt.fill_between(Nstep_list, min_dSB, max_dSB, color='red', alpha=.1)
    plt.plot(Nstep_list, [bestknown]*len(Nstep_list), '--k')
    plt.xlabel('Nstep')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig(folder + '\SB_nstep_Test.png')
    plt.show()


def test_BLIM():
    print("--------------------------------------")
    print("Test BLIM")
    print("--------------------------------------")
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folder = parent_dir + '\output\BLIM_test'
    file = os.path.join(parent_dir, "WK2000_1.txt")
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0, 0])
    nb_edges = int(info[0, 1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = hf.get_coeffs_from_array_MC(N, info)
    S = 1000
    tend = 3e-6
    dt = tend/S
    C = 1e-6
    G = 1e-1
    v_init = np.random.uniform(-1, 1, N)
    print('Changing k(t)')
    def changing_kt(t):
        kmin = 5.
        kmax = 20.
        cycle_duration = tend / 10
        return kmax if int(t // (cycle_duration/2)) % 2 == 0 else kmin
    v_optim, energies_ch, _, v_list_ch = BLIM.BLIM(J, v_init, dt, S, changing_kt, N, C, G, verbose=True)
    optim_energy = hf.compute_energy(J, h, np.sign(v_optim))
    print(f'Optimal energy changing k: {optim_energy}')

    print('Fixed k(t)')
    k_list = np.linspace([5.], [30.], 6)
    energies = []
    def constant_k(t, k):
        return k
    for k in k_list:
        const_k = lambda t: constant_k(t, k)
        v_optim, energy, times, v_list = BLIM.BLIM(J, v_init, dt, S, const_k, N, C, G, verbose=True)
        energies.append(energy)
        optim_energy = hf.compute_energy(J, h, np.sign(v_optim))
        print(f'Optimal energy constant k: {optim_energy}')

    hf.plot_energies({'changing k' : energies_ch, 'k = 5.': energies[0], 'k=10':energies[1], 'k=15':energies[2], 'k=20':energies[3], 'k=25':energies[4], 'k=30':energies[5]}, times, 'time',  filename=folder + '\energies_all.png')
    plt.figure()
    plt.imshow(np.sign(v_list_ch), interpolation='nearest')
    plt.title('Changing k')
    plt.show()

    plt.figure()
    plt.imshow(np.sign(v_list), interpolation='nearest')
    plt.title('constant k')
    plt.show()    

    

if __name__ == "__main__":
    if VERBOSE_PLOT:
        plt.ion()
    # problem1()
    # G1()
    k2000()
    #test_SB()
    # test_BLIM()

