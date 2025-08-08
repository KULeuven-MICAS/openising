import numpy as np

from ising.stages.model.ising import IsingModel

def find_cluster_Wolff(model:IsingModel, sigma:np.ndarray, start_node:int, P_add:float, max_size:int) -> np.ndarray:
    stack = [start_node]
    cluster = [start_node]

    coupling = model.J + model.J.T

    spins = np.arange(model.num_variables)
    for node in stack:
        neighbours = spins[(coupling[node, :] != 0.)]

        for neighbour in neighbours:
            if neighbour in cluster:
                continue
            
            rand_val = np.random.uniform(0, 1)
            if sigma[neighbour] == sigma[start_node] and rand_val < P_add:
                cluster.append(neighbour)
                stack.append(neighbour)
        
        stack.remove(node)
        if len(cluster) >= max_size:
            break
    return cluster