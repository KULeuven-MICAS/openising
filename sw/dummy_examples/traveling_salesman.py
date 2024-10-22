import openjij as oj
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def example1():
    W = {(0, 1): 10, (0, 2): 15, (0, 3): 25,
         (1, 2): 35, (1, 3): 15, (1, 0): 10,
         (2, 3): 10, (2, 0): 15, (2, 1): 35,
         (3, 0): 25, (3, 1): 15, (3, 2): 10}
    A = 40
    nb_cities = 4
    Q = {((i, time), (j, time_)):0 for i in range(nb_cities) for j in range(nb_cities) for time in range(nb_cities) for time_ in range(nb_cities)}
    #Q = add_vertex_constraint(nb_cities, Q)
    add_objective(nb_cities, Q, W)
    add_edge_constraint(Q, W, A)
    add_vertex_constraint(nb_cities, Q, A)
    add_time_constraint(nb_cities, Q, A)
    Q = simplify_Q(Q)

    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=500)
    print(response.first.sample)
    print(response.first.energy)
    plot_solution(response.first.sample, W, nb_cities)
    plot_energies(response.energies)
    
def example2():
    W = {(0 ,1) : 43, (0, 2) : 59, (0, 3) : 36, (0, 4) : 80, (0, 5) : 20, (0, 6) : 26, 
         (1, 0) : 12, (1, 2) : 99, (1, 3) : 81, (1, 4) : 50, (1, 5) : 19, (1, 6) : 70,
         (2, 0) : 84, (2, 1) : 17, (2, 3) : 25, (2, 4) : 18, (2, 5) : 99, (2, 6) : 33,
         (3, 0) : 97, (3, 1) : 97, (3, 2) : 38, (3, 4) : 31, (3, 5) : 49, (3, 6) : 66,
         (4, 0) : 41, (4, 1) : 17, (4, 2) : 23, (4, 3) : 29, (4, 5) : 79, (4, 6) : 69,
         (5, 0) : 30, (5, 1) : 11, (5, 2) : 42, (5, 3) : 35, (5, 4) : 100, (5, 6) : 36,
         (6, 0) : 73, (6, 1) : 13, (6, 2) : 35, (6, 3) : 68, (6, 4) : 74, (6, 5) : 51}
    A = W[max(W, key=W.get)] + 10
    nb_cities = 7
    Q = {((i, time), (j, time_)):0 for i in range(nb_cities) for j in range(nb_cities) for time in range(nb_cities) for time_ in range(nb_cities)}
    #Q = add_vertex_constraint(nb_cities, Q)
    add_objective(nb_cities, Q, W)
    add_edge_constraint(Q, W, A)
    add_vertex_constraint(nb_cities, Q, A)
    add_time_constraint(nb_cities, Q, A)
    Q = simplify_Q(Q)

    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=500)
    print(response.first.sample)
    print(response.energies)
    plot_solution(response.first.sample, W, nb_cities)
    plot_energies(response.energies)


def example3():
    "Example with sparse graph"
    W = {(0, 3): 10, (3, 5): 10, (5, 11): 10, (11, 9): 10, (9, 6): 10, (6, 4): 10, (4, 7): 10, (7, 12): 10, (12, 8): 10, (8, 10): 10, (10, 2): 10, (2, 1): 10, (1, 0): 10,
         (3, 12): 20, (3, 4): 25, (0, 8): 30, (2, 0): 85, (4, 5): 35, (4, 8): 49,
         (5, 7): 68, (7, 10): 94, (9, 5): 162, (11, 4): 37, (11, 6): 185, (12, 5): 62
    }

    A = W[max(W, key=W.get)] + 20
    nb_cities = 13
    Q = {((i, time), (j, time_)):0 for i in range(nb_cities) for j in range(nb_cities) for time in range(nb_cities) for time_ in range(nb_cities)}
    #Q = add_vertex_constraint(nb_cities, Q)
    add_objective(nb_cities, Q, W)
    add_edge_constraint(Q, W, A)
    add_vertex_constraint(nb_cities, Q, A)
    add_time_constraint(nb_cities, Q, A)
    Q = simplify_Q(Q)

    sampler = oj.SASampler()
    response = sampler.sample_qubo(Q, num_reads=500)
    print(response.first.sample)
    print(response.energies)
    plot_solution(response.first.sample, W, nb_cities)
    plot_energies(response.energies)


def simplify_Q(Q):
    return {x:y for x, y in Q.items() if y!=0}


def add_vertex_constraint(nb_cities, Q, A):
    """
    Adds the constraint that every vertex must be visited once.

    Args:
        nb_cities: the amount of cities that are in the graph.
        Q: the QUBO matrix in dictionary form.
        A: the constraint constant.
    """
    for city in range(nb_cities):
        for time in range(nb_cities):
            Q[((city, time), (city, time))] += -A
            for _time in range(time+1,nb_cities):
                Q[((city, time), (city, _time))] += 2*A


def add_time_constraint(nb_cities, Q, A):
    """
    Adds the constraint that at every time step a node should be visited.

    Args:
        nb_cities: the amount of cities in the graph
        Q: the QUBO matrix in dictionary form
        A: the constraint constant
    """
    for time in range(nb_cities):
        for city in range(nb_cities):
            Q[((city, time), (city, time))] += -A
            for _city in range(city+1, nb_cities):
                Q[((city, time), (_city, time))] += 2*A


def add_edge_constraint(Q, W, A):

    for key in Q.keys():
        ((i, time), (j, time_)) = key
        if ((i, j) not in W.keys()) and (time_-time == 1) and (i != j):
            Q[key] += A


def add_objective(nb_cities, Q, W):
    for (edge, weight) in W.items():
        (i, j) = edge
        for time in range(nb_cities):
            if time + 1 == nb_cities:
                Q[((i, time), (j, 0))] += weight
            else:
                Q[((i, time), (j, time+1))] += weight


def plot_solution(response, W, nb_cities):
    G = nx.DiGraph()
    route = [None]*nb_cities
    total_weight_route = 0
    weight_edges = {}
    for (node, bit) in response.items():
        if bit:
            G.add_node(node[0])
            route[node[1]] = node[0]

    first = True
    first_node = []
    for i in range(nb_cities):
        node = route[i]
        if i != nb_cities-1:
            next_node = route[i+1]
        else: 
            next_node = route[0]
        if first:
            first_node.append(node)
        G.add_edge(node, next_node)
        weight_edges[(node, next_node)] = W[(node, next_node)]
        total_weight_route += W[(node, next_node)]
    pos = nx.spring_layout(G)
    plt.subplots()
    nx.draw_networkx(G, pos)
    nx.draw_networkx_nodes(G, pos, nodelist=first_node, node_color="tab:red")
    nx.draw_networkx_nodes(G, pos, nodelist=[node for node in route if node != first_node[0]], node_color='tab:blue')
    nx.draw_networkx_edge_labels(G, pos, weight_edges)
    plt.title(f"The total weight is {total_weight_route}.")
    plt.show()

def plot_energies(energies):
    plt.figure()
    plt.hist(energies, bins=30, edgecolor='black')
    plt.xlabel(r'Energy', fontsize=15)
    plt.ylabel(r'Frequency', fontsize=15)
    plt.title(f"The minimum energy is: {min(energies)}")
    plt.show()

if __name__ == '__main__':
    #example1()
    #example2()
    example3()
