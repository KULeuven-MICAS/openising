import SCA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red')
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()


def compute_energy(J, h, sigma):
    energy = 0.
    N = np.size(sigma)[0]
    for i in range(N):
        energy -= h[i]*sigma[i]
        for j in range(i+1, N):
            energy -= J[i, j]*sigma[i]*sigma[j]
    return energy


def problem1():
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
    plt.show()

    h = np.zeros((N,))
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            if G.has_edge(i, j):
                J[i, j] = -1
    sigma = np.random.randint(1, size=(N,))
    for i in range(N):
        if sigma[i]==0:
            sigma[i] = -1
    S = 100
    q_init = 2.0
    T_init = 50.0
    r_q = (3.0/q_init)**(1/(S-1))
    r_t = (5/T_init)**(1/(S-1))
    sigma_optim = SCA.SCA(sigma, J, h, S, q_init, T_init, r_q, r_t)
    energy = compute_energy(J, h, sigma_optim)
    print(f"The optimal energy is: {energy}")
    print(f"The optimal state: {sigma_optim}")
    plot_solution(sigma_optim, G)
    

if __name__=="__main__":
    problem1()