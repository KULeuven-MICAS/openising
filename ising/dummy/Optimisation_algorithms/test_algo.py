from SCA import SCA, compute_energy
from SA import SA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import openjij as oj
#from  ising.model import BinaryQuadraticModel

VERBOSE = False
VERBOSE_SOLVER = True
VERBOSE_PLOT = False


def add_edges_graph(graph, states, G_orig):
    red_nodes = []
    blue_nodes = []
    labels = {}
    for i in range(len(states)):
        graph.add_node(i)
        labels[i] = i
        if states[i] == 1:
            red_nodes.append(i)
        else:
            blue_nodes.append(i)
        for j in range(len(states)):
            if i != j and states[i] == states[j] and G_orig.has_edge(i, j):
                graph.add_edge(i, j)
    return red_nodes, blue_nodes, labels


def plot_solution(state, G_orig, solver):     
    G = nx.Graph()
    red_nodes, blue_nodes, labels = add_edges_graph(G, state, G_orig)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red')
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.title(solver)


def get_random_s(N):
    sigma = np.random.choice([-1, 1], N)
    return sigma


def get_coeffs_from_array(N, data):
    J = np.zeros((N, N))
    h = np.zeros((N,))
    for row in data:
        i, j, weight = int(row[0])-1, int(row[1])-1, row[2]
        J[i, j] = -weight
        J[j, i] = -weight
    return J, h

def run_solver(solver, s_init, J, h, S, T, r_t, N, G=None, q=0, r_q=0, dir='.'):
    if solver == 'SCA':
        sigma_optim, energies = SCA(s_init=s_init, J=J, h_init=h, S=S, q_init=q, T_init=T, r_q=r_q, r_t=r_t, verbose=VERBOSE_SOLVER)
    else:
        sigma_optim, energies = SA(T, r_t, S, J, h, s_init, verbose=VERBOSE_SOLVER)
    energy = -1/2*np.inner(sigma_optim.T, np.inner(J, sigma_optim)) - np.inner(h.T, sigma_optim)
    print(f"The optimal energy of {solver} is: {energy}")
    if N <=20:
        print(f"The optimal state of {solver}: {sigma_optim}")
        plot_solution(sigma_optim, G, solver)
    plt.figure()
    plt.plot(np.array(list(range(S))), energies)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.title(f"Energy {solver}")
    plt.savefig(f"{dir}\Energy_{solver}.png")
    return sigma_optim, energies


def compute_rx(init, end, S):
    return (end/init)**(1/(S-1))

def problem1():
    print("--------------------------------------")
    print("Not-fully connected graph with 8 nodes")
    print("--------------------------------------")
    data = np.array([[0, 1, 1.],
                     [0, 3, 1.],
                     [1, 2, 1.],
                     [2, 3, 1.],
                     [2, 7, 1.],
                     [3, 6, 1.],
                     [4, 5, 1.],
                     [4, 6, 1.],
                     [4, 7, 1.],
                     [5, 6, 1.],
                     [6, 7, 1.],])
    
    N=8    
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    for row in data:
        i, j = row[0], row[1]
        G.add_edge(i, j)
    nx.draw_networkx(G)
    folder = '..\output'

    J, h = get_coeffs_from_array(N, data)
    nb_runs = 15
    S = 500
    T = 50.
    q = 1.
    T_end = 0.05
    q_end = 3.
    r_t = compute_rx(T, T_end, S)
    r_q = compute_rx(q, q_end, S)
    s_init = get_random_s(N)
    print(f"Initial sigma: {s_init}")
    
    mat = np.diag(h) - 1/2*J
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype='SPIN')
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=nb_runs)
    plt.figure()
    plt.hist(response.energies, 10)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('SA OpenJij energy outline')
    plt.savefig(folder + '/histogram_sa_openjij.png')

    print(f"Solution of OpenJij: {response.first.sample}")
    print(f"Optimal energy of OpenJij:  {response.first.energy}")
    plot_solution(response.first.sample, G, 'SA OpenJij')

    energy_sca = []
    energy_sa = []
    for i in range(nb_runs):
        sigma_sca, energies_sca = run_solver("SCA", s_init, J, h, S, T, r_t, N, G=G, q=q, r_q=r_q, dir=folder)
        energy_sca.append(energies_sca[-1])
        sigma_sca = np.copy(sigma_sca)
        sigma_SA, energies_sa = run_solver("SA", s_init, J, h, S, T, r_t, N=N, G=G, dir=folder)
        energy_sa.append(energies_sa[-1])
    plt.figure()
    plt.hist(energy_sca, 10)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('SCA energy outline')
    plt.savefig(folder + '/histogram_sca.png')
    plt.figure()
    plt.hist(energy_sa, 10)
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('SA energy outline')
    plt.savefig(folder + '/histogram_sa.png')
    

    plt.figure()
    plt.plot(sigma_sca, '^r', label='SCA')
    plt.plot(sigma_SA, "+b", label='SA')
    plt.plot(list(response.first.sample.values()), '.g', label='SA openjij')
    plt.xlabel('spin')
    plt.ylabel('spin value')
    plt.title('Spin differences SCA, SA and OpenJij')
    plt.legend()
    plt.savefig(folder + '\state_all.png')

    plt.figure()
    plt.plot(list(range(S)), energies_sca, label='SCA')
    plt.plot(list(range(S)), energies_sa, label='SA')
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.title('Energy comparison between SCA and SA')
    plt.savefig(folder + '\energies_all.png')


def importance_hyperparameters_SCA():
    print("--------------------------------------")
    print("Influence of hyperparameters of SCA")
    print("--------------------------------------")

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    file = os.path.join(parent_dir, 'G1.txt')
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0,0])
    nb_edges = int(info[0,1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = get_coeffs_from_array(N, info)
    sigma = get_random_s(N)


    print('Influence of r_q')
    print("--------------------------------------")
    S = 560
    q_init = 2.
    T_init = 50.0
    T_end = 4.
    q_fin = [3., 4., 5., 6.]
    r_t = (T_end/T_init)**(1/(S-1))

    for q_end in q_fin:
        r_q = (q_end/q_init)**(1/(S-1))
        print(f"Final q: {q_end}, increase rate: {r_q}")
        sigma_SCA, energy_SCA = SCA(s_init=sigma, J=J, h_init=h, S=S, q_init=q_init, T_init=T_init, r_q=r_q, r_t=r_t)
        energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(h.T, sigma_SCA)
        print(f"The optimal energy of SCA is: {energy}")

        plt.plot(np.array(list(range(S))), energy_SCA)
        plt.xlabel("iteration")
        plt.ylabel("Energy")
        plt.savefig(f'Energy_SCA_rq{str(r_q)}.png')
        plt.show()
    
    print('Influence of r_T')
    print("--------------------------------------")
    S = 560
    q_init = 2.
    T_init = 50.0
    T_fin = [4., 5., 6.]
    q_end = 4.
    r_q = (q_end/q_init)**(1/(S-1))

    for T_end in T_fin:
        r_t = (T_end/T_init)**(1/(S-1))
        print(f"Final T: {q_end}, decrease rate: {r_q}")
        sigma_SCA, energy_SCA = SCA(s_init=sigma, J=J, h_init=h, S=S, q_init=q_init, T_init=T_init, r_q=r_q, r_t=r_t)
        energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(h.T, sigma_SCA)
        print(f"The optimal energy of SCA is: {energy}")

        plt.plot(np.array(list(range(S))), energy_SCA)
        plt.xlabel("iteration")
        plt.ylabel("Energy")
        plt.savefig(f'Energy_SCA_rT{str(r_t)}.png')
        plt.show()
    
    print('Influence of S')
    print("--------------------------------------")
    S_all = [560, 760, 960, 1160, 1360]
    q_init = 2.
    T_init = 50.0
    T_end = 4.
    q_end = 4.
    r_q = (q_end/q_init)**(1/(S-1))
    r_t = (T_end/T_init)**(1/(S-1))


    for S in S_all:
        print(f"Iterations S: {S}, decrease rate: {r_q}")
        sigma_SCA, energy_SCA = SCA(s_init=sigma, J=J, h_init=h, S=S, q_init=q_init, T_init=T_init, r_q=r_q, r_t=r_t)
        energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(h.T, sigma_SCA)
        print(f"The optimal energy of SCA is: {energy}")

        plt.plot(np.array(list(range(S))), energy_SCA)
        plt.xlabel("iteration")
        plt.ylabel("Energy")
        plt.savefig(f'Energy_SCA_S{str(S)}')
        plt.show()

def problem2():
    print("--------------------------------------")
    print("Not-fully connected graph from G1 benchmark")
    print("--------------------------------------")
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    file = os.path.join(parent_dir, 'G1.txt')
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0,0])
    nb_edges = int(info[0,1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = get_coeffs_from_array(N, info)
    s_init = get_random_s(N)

    folder = '../output/algo_problem2'
    S = 10000
    q = 1.
    T = 50.0
    T_end = 0.05
    q_end = 5.
    r_q = compute_rx(q, q_end, S)
    r_t = compute_rx(T, T_end, S)
    
    sigma_sca, energies_sca = run_solver("SCA", s_init, J, h, S, T, r_t, N, q=q, r_q=r_q, dir=folder)
    sigma_sca = np.copy(sigma_sca)
    sigma_SA, energies_sa = run_solver("SA", s_init, J, h, S, T, r_t, N=N, dir=folder)

    plt.plot(np.array(list(range(S))), energies_sca)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.savefig(folder + '/Energy_SCA.png')


    mat = np.diag(h) - 1/2*J
    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(mat, vartype='SPIN')
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=500)
    print(response.first.energy)

    plt.figure()
    plt.plot(energies_sa)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.savefig(folder + '/Energy_SA.png')

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
    Q = np.array([[ 2, -2, -2,  0,  0,  0,  0,  0,  0,  0],
                  [ 0,  2, -2,  0,  0,  0,  0,  0,  0,  0],
                  [ 0,  0,  3, -2,  0,  0,  0,  0,  0,  0],
                  [ 0,  0,  0,  3, -2, -2,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  3, -2, -2,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  3, -2,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  0,  3, -2,  0,  0],
                  [ 0,  0,  0,  0,  0,  0,  0,  3, -2, -2],
                  [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2],
                  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  2]])
    
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



if __name__=="__main__":
    if VERBOSE_PLOT:
        plt.ion()
    #problem1()
    #importance_hyperparameters_SCA()
    problem2()
    #problem3()