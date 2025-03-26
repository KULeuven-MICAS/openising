import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def make_J(W, N, A, B, C):
    J = np.zeros((N**2, N**2))
    for i in range(N):
        for k in range(N):
            for j in range(N):
                for l in range(N):
                    if j==(i+1) or i == (j+1) or (i==0 and j==N-1) or (i == N-1 and j==0) and (k != l):
                        J[i*k, j*l] -= A/8*W[k, l]
                    elif (i==j) and (k != l):
                        J[i*k, j*l] -= B/4
                    elif (k==l) and (i != j):
                        J[i*k, j*l] -= C/4
                    elif (k==l) and (i==j):
                        J[i*k, j*l] -= (B+C)/4
    return J

def make_h(W, N, A, B, C):
    h = np.zeros((N**2,))
    for i in range(N):
        for k in range(N):
            h[i*k] -= (N-2)*(B+C)/2
            for l in range(N):
                if l != k:
                    h[i*k] -= A/2*W[k, l]
    return h

N = 3
W = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])
G = nx.DiGraph()
G.add_nodes_from([1, 2, 3])
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 1)
edge_labels = {(1, 2): 'W12',
               (2, 3): 'W23',
               (3, 1): 'W31'}

plt.figure()
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()

J = make_J(W, N, A=8, B=5, C=3)
print(J)
h = make_h(W, N, A=8, B=5, C=3)

nodes = list()
edges = list()
pos = dict()
for i in range(N):
    for k in range(N):
        nodes.append(f'{i}, {k}')
        pos[f'{i}, {k}'] = (i, k)
        for j in range(N):
            for l in range(N):
                if J[i*k, j*l] != 0.:
                    edges.append((f'{i}, {k}', f'{j}, {l}', J[i*k, j*l]))
Gnew = nx.Graph()
Gnew.add_nodes_from(nodes)
Gnew.add_weighted_edges_from(edges)

plt.figure()
nx.draw_networkx(Gnew, pos)
nx.draw_networkx_nodes(Gnew, pos, nodelist=nodes, node_size=400)
nx.draw_networkx_edges(Gnew, pos, edgelist=edges, connectionstyle="arc3,rad=0.1", arrows=True)
weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
plt.axis('off')
plt.show()