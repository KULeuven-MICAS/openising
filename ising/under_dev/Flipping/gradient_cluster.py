import numpy as np

from ising.stages.model.ising import IsingModel

def find_cluster_gradient(model:IsingModel, sigma:np.ndarray, max_size:int, threshold:float)->np.ndarray:
    coupling = model.J + model.J.T
    gradient = coupling @ sigma + model.h
    gradient /= np.max(gradient)
    available_nodes = np.where(gradient >= threshold, np.arange(len(sigma)), -1)
    if len(available_nodes[available_nodes >= 0]) < max_size:
        current_size = len(available_nodes[available_nodes >= 0])
        ind_unavailable_nodes = np.where(available_nodes < 0)[0]
        chosen_nodes = np.array([], dtype=int)
        while len(chosen_nodes) < max_size - current_size:
            chosen_nodes = np.unique(np.append(chosen_nodes, np.random.choice(ind_unavailable_nodes, (max_size - current_size-len(chosen_nodes),))))
        available_nodes[chosen_nodes] = np.arange(len(sigma))[chosen_nodes]
        cluster = available_nodes[available_nodes >= 0]
    else:
        cluster = np.array([], dtype=int)
        while len(cluster) < max_size:
            cluster = np.unique(np.append(cluster, np.random.choice(available_nodes[available_nodes >= 0], size=(max_size-len(cluster),))))
    # LOGGER.info(f"cluster: {cluster}")
    return cluster

def find_cluster_gradient_largest(model:IsingModel, sigma:np.ndarray, max_size:int) -> np.ndarray:
    coupling = model.J + model.J.T
    gradient = np.abs(coupling @ sigma + model.h)
    cluster = np.argpartition(gradient, -max_size)[-max_size:]
    return cluster
