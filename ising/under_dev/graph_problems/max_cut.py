import openjij as oj
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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


def plot_solutions(response, G_orig):     
    for state in response.states:
        print(state)
        G = nx.Graph()
        red_nodes, blue_nodes, labels = add_edges_graph(G, state, G_orig)
        pos = nx.spring_layout(G, seed=3113794652)
        nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red')
        nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue')
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()

def example1():
    print('--------------------------------')
    print("Example 1")
    print('--------------------------------')

    print('--------------------------------')
    print("Visualization of the graph")
    print('--------------------------------')

    G_orig = nx.Graph()
    G_orig.add_edge(0, 1)
    G_orig.add_edge(1, 3)
    G_orig.add_edge(0, 2)
    G_orig.add_edge(2, 3)
    G_orig.add_edge(2, 4)
    G_orig.add_edge(3, 4)
    nx.draw_networkx(G_orig)
    plt.show()

    print('--------------------------------')
    print("Solving Max-cut problem")
    print('--------------------------------')
    Q = {(0, 0): -2, (0, 1): 1, (0, 2): 1, 
        (1, 0): 1, (1, 1): -2, (1, 3): 1, 
        (2, 0): 1, (2, 2): -3, (2, 3): 1, (2, 4): 1,
        (3, 1): 1, (3, 2): 1, (3, 3): -3, (3, 4): 1,
        (4, 2): 1, (4, 3): 1, (4, 4): -2}

    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=5)

    print(response.states)
    print(response.energies)
    plot_solutions(response, G_orig)

def example2():
    print('--------------------------------')
    print("Example 2")
    print('--------------------------------')

    G_orig = nx.Graph()
    G_orig.add_edge(0, 1)
    G_orig.add_edge(0, 2)
    G_orig.add_edge(1, 4)
    G_orig.add_edge(1, 3)
    G_orig.add_edge(2, 5)
    G_orig.add_edge(3, 5)
    G_orig.add_edge(4, 5)
    nx.draw_networkx(G_orig)
    plt.show()

    Q = -np.array([[2, -1, -1, 0, 0, 0],
                [-1, 3, 0, -1, -1, 0],
                [-1, 0, 2, 0, 0, -1],
                [0, -1, 0, 2, 0, -1],
                [0, -1, 0, 0, 2, -1],
                [0, 0, -1, -1, -1, 3]])

    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(Q, vartype='BINARY')
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=5)

    print(response.states)
    print(response.energies)
    plot_solutions(response, G_orig)

def example3():
    print('--------------------------------')
    print("Example 3")
    print('--------------------------------')

    G_orig = nx.Graph()
    G_orig.add_edge(1, 2)
    G_orig.add_edge(1, 3)
    G_orig.add_edge(1, 4)
    G_orig.add_edge(2, 5)
    G_orig.add_edge(2, 6)
    G_orig.add_edge(3, 6)
    G_orig.add_edge(3, 7)
    G_orig.add_edge(4, 7)
    G_orig.add_edge(5, 8)
    G_orig.add_edge(6, 8)
    G_orig.add_edge(7, 8)
    G_orig.add_edge(4, 5)
    G_orig.add_edge(8, 9)
    G_orig.add_edge(6, 9)
    G_orig.add_edge(7, 10)
    nx.draw_networkx(G_orig)
    plt.show()

    Q = -np.array([[3, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                [-1, 3, 0, 0, -1, -1, 0, 0, 0, 0],
                [-1, 0, 3, 0, 0, -1, -1, 0, 0, 0],
                [-1, 0, 0, 3, -1, 0, -1, 0, 0, 0],
                [0, -1, 0, -1, 3, 0, 0, -1, 0, 0],
                [0, -1, -1, 0, 0, 4, 0, -1, -1, 0],
                [0, 0, -1, -1, 0, 0, 4, -1, 0, -1],
                [0, 0, 0, 0, -1, -1, -1, 4, -1, 0],
                [0, 0, 0, 0, 0, -1, 0, -1, 2, 0],
                [0, 0, 0, 0, 0, 0, -1, 0, 0, 1]])

    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(Q, vartype='BINARY')
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=5)

    print(response.states)
    print(response.energies)
    plot_solutions(response, G_orig)

if __name__ == '__main__':
    #example1()
    #example2()
    example3()