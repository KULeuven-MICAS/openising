import numpy as np

from ising.stages.model.ising import IsingModel

def find_cluster_gradient(model:IsingModel, sigma:np.ndarray, max_size:int, threshold:float)->np.ndarray:
    coupling = model.J + model.J.T
    gradient = coupling @ sigma + model.h
    gradient /= np.max(gradient)
    available_nodes = np.where(np.abs(gradient) >= threshold, np.arange(len(sigma)), -1)
    if len(available_nodes[available_nodes >= 0]) < max_size:
        current_size = len(available_nodes[available_nodes >= 0])
        ind_unavailable_nodes = np.where(available_nodes < 0)[0]
        chosen_nodes = np.random.choice(ind_unavailable_nodes, (max_size - current_size,))
        available_nodes[chosen_nodes] = np.arange(len(available_nodes))[chosen_nodes]
    cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(max_size,))
    return cluster

def find_cluster_gradient_largest(model:IsingModel, sigma:np.ndarray, max_size:int) -> np.ndarray:
    coupling = model.J + model.J.T
    gradient = np.abs(coupling @ sigma + model.h)
    cluster = np.argpartition(gradient, -max_size)[-max_size:]
    return cluster