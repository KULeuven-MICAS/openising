from SCA import SCA, compute_energy
import SA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os


def add_edges_graph(graph, states, G_orig):
    red_nodes = []
    blue_nodes = []
    labels = {}
    for i in range(len(states)):
        graph.add_node(i)
        labels[i] = i + 1
        if states[i] == 1:
            red_nodes.append(i)
        else:
            blue_nodes.append(i)
        for j in range(len(states)):
            if i != j and states[i] == states[j] and G_orig.has_edge(i, j):
                graph.add_edge(i, j)
    return red_nodes, blue_nodes, labels


def plot_solution(state, G_orig):     
    print(state)
    G = nx.Graph()
    red_nodes, blue_nodes, labels = add_edges_graph(G, state, G_orig)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red')
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()


def get_random_s(N):
    sigma = np.random.randint(0, 2, size=(N,))
    for i in range(N):
        if sigma[i]==0:
            sigma[i] = -1
    return sigma


def get_coeffs_from_array(N, data):
    J = np.zeros((N, N))
    h = np.zeros((N,))
    for row in data:
        i, j, weight = int(row[0])-1, int(row[1])-1, row[2]
        if i < j:
            J[i,j] = -weight
        else:
            J[j, i] = -weight
    return J, h


def problem1(verbose=False):
    print("--------------------------------------")
    print("Not-fully connected graph with 5 nodes")
    print("--------------------------------------")
    N = 5
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(3, 4)
    nx.draw_networkx(G)
    if verbose:
        plt.show()

    h = np.zeros((N,))
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            if G.has_edge(i, j):
                J[i, j] = -1
    sigma = get_random_s(N)
    
    S = 50
    q_init = 8.0
    T_init = 50.0
    r_q = (12.0/q_init)**(1/(S-1))
    r_t = (5/T_init)**(1/(S-1))
    sigma_SCA, energy_SCA = SCA(s_init=sigma, J=J, h_init=h, S=S, q_init=q_init, T_init=T_init, r_q=r_q, r_t=r_t)
    energy = -np.inner(sigma_SCA.T, np.inner(J, sigma_SCA)) - np.inner(h.T, sigma_SCA)
    print(f"The optimal energy of SCA is: {energy}")
    print(f"The optimal state of SCA: {sigma_SCA}")
    plot_solution(sigma_SCA, G)

    plt.plot(np.array(list(range(S))), energy_SCA)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.show()

    T_min = 0.05
    sigma_SA, energy_SA = SA.SA(T_init, r_t, T_min, J, h, sigma)
    energy = -np.inner(sigma_SA.T, np.inner(J, sigma_SA)) - np.inner(h.T, sigma_SA)
    print(f"The optimal energy of SA is: {energy}")
    print(f"The optimal state of SA: {sigma_SA}")
    plot_solution(sigma_SA, G)

    plt.plot(np.array(list(range(len(energy_SA)))), energy_SA)
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    plt.show()


def problem2():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    file = os.path.join(parent_dir, 'G1.txt')
    info = np.genfromtxt(file, delimiter=" ")
    N = int(info[0,0])
    nb_edges = int(info[0,1])
    print(f"Graph with {N} nodes and {nb_edges} edges")
    info = info[1:, :]
    J, h = get_coeffs_from_array(N, info)
    sigma = get_random_s(N)


if __name__=="__main__":
    #problem1()
    problem2()