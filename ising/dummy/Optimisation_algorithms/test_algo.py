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
# from  ising.model import BinaryQuadraticModel

VERBOSE = False
VERBOSE_SOLVER = False
VERBOSE_PLOT = False


def run_solver(solver:str, s_init:np.ndarray, J:np.ndarray, h:np.ndarray, S:int, N:int, dir:str=".", G=None, **hyperparameters):
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
            q=hyperparameters['q'],
            T=hyperparameters['T'],
            r_q=hyperparameters['r_q'],
            r_t=hyperparameters['r_t'],
            verbose=VERBOSE_SOLVER,
        )
    elif solver[1:] == "SB":
        x_init = np.multiply(
            np.random.uniform(low=0.0, high=0.5, size=np.shape(s_init)), s_init
        )
        y_init = np.random.uniform(-0.5, 0.5, size=np.shape(s_init))
        if solver[0] == 'b':
            sigma_optim, energies, times = ballisticSB(
                h=h,
                J=J,
                x_init=x_init,
                y_init=y_init,
                dt=hyperparameters['dt'],
                Nstep=S,
                a0=hyperparameters['a0'],
                c0=hyperparameters['c0'],
                at=hyperparameters['at'],
                verbose=VERBOSE_SOLVER,
            )
        else:
            sigma_optim, energies, times = discreteSB(
                h=h,
                J=J,
                x_init=x_init,
                y_init=y_init,
                dt=hyperparameters['dt'],
                Nstep=S,
                a0=hyperparameters['a0'],
                c0=hyperparameters['c0'],
                at=hyperparameters['at'],
                verbose=VERBOSE_SOLVER,
            )

    else:
        sigma_optim, energies = SA(
            T=hyperparameters['T'],
            r_T=hyperparameters['r_t'],
            S=S,
            J=J,
            h=h,
            sigma=s_init,
            verbose=VERBOSE_SOLVER,
        )
    energy = -np.inner(sigma_optim.T, np.inner(J, sigma_optim)) - np.inner(
        h.T, sigma_optim
    )
    print(f"The optimal energy of {solver} is: {energy}")
    if N <= 20:
        print(f"The optimal state of {solver}: {sigma_optim}")
        hf.plot_solution(sigma_optim, G, solver)
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

    print('Setting hyperparameters')
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
    a0 = 1.
    c0 = 0.5/(math.sqrt(N)*math.sqrt(np.sum(np.power(J, 2))/(N*(N-1))))
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
            solver="SCA", s_init=s_init, J=J, h=h, S=S, N=N, G=G, dir=folder, q=q, r_q=r_q, T=T, r_t=r_t
        )
        energy_sca.append(energies_sca[-1])
        sigma_SA, energies_sa = run_solver(
            solver="SA", s_init=s_init, J=J, h=h, S=S, N=N, G=G, dir=folder, T=T, r_t=r_t
        )
        energy_sa.append(energies_sa[-1])
        sigma_bsb, energies_bsb = run_solver(solver='bSB', s_init=s_init, J=J, h=h, S=S, N=N, G=G, dir=folder, a0=a0, c0=c0, at=at, dt=dt)
        energy_bsb.append(energies_bsb[-1])
        sigma_dsb, energies_dsb = run_solver(solver='dSB', s_init=s_init, J=J, h=h, S=S, N=N, G=G, dir=folder, a0=a0, c0=c0, at=at, dt=dt)
        energy_dsb.append(energies_dsb[-1])

    hf.plot_energy_dist(energy_sca, "SCA", folder + "\histogram_sca.png")
    hf.plot_energy_dist(energy_sa, "SA", folder + "\histogram_sa.png")
    hf.plot_energy_dist(energy_bsb, 'bSB', folder+'\histogram_bSB.png')
    hf.plot_energy_dist(energies_dsb, 'dSB', folder+'\histogram_dSB.png')

    energies = {"SCA": energies_sca, "SA": energies_sa, 'bSB': energies_bsb, 'dSB': energies_dsb}
    hf.plot_energies(energies, S, folder + "\energies_all.png")


def importance_hyperparameters_SCA():
    print("--------------------------------------")
    print("Influence of hyperparameters of SCA")
    print("--------------------------------------")

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    file = os.path.join(parent_dir, "G1.txt")
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0, 0])
    nb_edges = int(info[0, 1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = hf.get_coeffs_from_array_MC(N, info)
    sigma = hf.get_random_s(N)

    print("Influence of r_q")
    print("--------------------------------------")
    S = 560
    q_init = 2.0
    T_init = 50.0
    T_end = 4.0
    q_fin = [3.0, 4.0, 5.0, 6.0]
    r_t = hf.compute_rx(T_init, T_end, S)

    for q_end in q_fin:
        r_q = (q_end / q_init) ** (1 / (S - 1))
        print(f"Final q: {q_end}, increase rate: {r_q}")
        sigma_SCA, energy_SCA = run_solver(
            "SCA",
            sigma,
            J,
            h,
            S,
            T_init,
            r_t,
        )
        energy = hf.compute_energy(J, h, sigma_SCA)
        print(f"The optimal energy of SCA is: {energy}")

        hf.plot_energies(
            {"SCA": energy_SCA},
            S,
            f"{parent_dir}\output\hyperparameter_test\Energy_SCA_rq{str(r_q)}.png",
        )

    print("Influence of r_T")
    print("--------------------------------------")
    S = 560
    q_init = 2.0
    T_init = 50.0
    T_fin = [4.0, 5.0, 6.0]
    q_end = 4.0
    r_q = (q_end / q_init) ** (1 / (S - 1))

    for T_end in T_fin:
        r_t = (T_end / T_init) ** (1 / (S - 1))
        print(f"Final T: {q_end}, decrease rate: {r_q}")
        sigma_SCA, energy_SCA = SCA(
            s_init=sigma,
            J=J,
            h_init=h,
            S=S,
            q_init=q_init,
            T_init=T_init,
            r_q=r_q,
            r_t=r_t,
        )
        energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(
            h.T, sigma_SCA
        )
        print(f"The optimal energy of SCA is: {energy}")

        hf.plot_energies(
            {"SCA": energy_SCA},
            S,
            f"{parent_dir}\output\hyperparameter_test\Energy_SCA_rT{str(r_t)}.png",
        )

    print("Influence of S")
    print("--------------------------------------")
    S_all = [560, 760, 960, 1160, 1360]
    q_init = 2.0
    T_init = 50.0
    T_end = 4.0
    q_end = 4.0
    r_q = (q_end / q_init) ** (1 / (S - 1))
    r_t = (T_end / T_init) ** (1 / (S - 1))

    for S in S_all:
        print(f"Iterations S: {S}, decrease rate: {r_q}")
        sigma_SCA, energy_SCA = SCA(
            s_init=sigma,
            J=J,
            h_init=h,
            S=S,
            q_init=q_init,
            T_init=T_init,
            r_q=r_q,
            r_t=r_t,
        )
        energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(
            h.T, sigma_SCA
        )
        print(f"The optimal energy of SCA is: {energy}")

        hf.plot_energies(
            {"SCA": energy_SCA},
            S,
            f"{parent_dir}\output\hyperparameter_test\Energy_SCA_S{str(S)}.png",
        )


def problem2():
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

    sigma_sca, energies_sca = run_solver(
        "SCA", s_init, J, h, S, T, r_t, N, q=q, r_q=r_q, dir=folder
    )
    sigma_sca = np.copy(sigma_sca)
    sigma_SA, energies_sa = run_solver("SA", s_init, J, h, S, T, r_t, N=N, dir=folder)

    plt.plot(np.array(list(range(S))), energies_sca)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.savefig(folder + "/Energy_SCA.png")

    mat = np.diag(h) - J
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype="SPIN")
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=500)
    print(response.first.energy)

    plt.figure()
    plt.plot(energies_sa)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.savefig(folder + "/Energy_SA.png")

    plt.figure()
    plt.plot(energies_sca)
    plt.plot(energies_sa)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.legend(["SCA", "SA"])
    plt.savefig(folder + "/Energy_comparison.png")


def problem3():
    print("--------------------------------------")
    print("Test with bqmpy code")
    print("--------------------------------------")
    # Q = np.array(
    #     [
    #         [2, -2, -2, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 2, -2, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 3, -2, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 3, -2, -2, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 3, -2, -2, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 3, -2, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 3, -2, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 3, -2, -2],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 2, -2],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    #     ]
    # )

    # bqm = bqmpy.BinaryQuadraticModel.from_qubo(-Q)
    # h, J = bqm.to_ising()
    # N = bqm.num_variables
    # sigma = get_random_s(N)
    # T = 50
    # S = 200
    # r_t = (0.05/T)**(1/(S-1))
    # print(f"Hyperparameters: initial temperature {T}, iteration {S}, temperature decrease {r_t}")
    # sigma_SA, energies_SA = SA(T, r_t, S, J, h, sigma)

    # plt.figure()
    # plt.plot(energies_SA)
    # plt.xlabel("iteration")
    # plt.ylabel("Energy")
    # plt.savefig('Energy_SA')
    # plt.show()


if __name__ == "__main__":
    if VERBOSE_PLOT:
        plt.ion()
    problem1()
    # importance_hyperparameters_SCA()
    # problem2()
    # problem3()
