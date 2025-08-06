import numpy as np

from ising.stages.model.ising import IsingModel
from ising.solvers.Multiplicative import Multiplicative
from ising.flow import LOGGER

def find_frozen_gradient_cluster(model:IsingModel, sigma:np.ndarray, nb_seeds:int) ->np.ndarray:
    coupling = model.J + model.J.T
    gradient = -coupling @ sigma - model.h
    freeze_score = np.abs(gradient) / np.max(np.abs(gradient))

    threshold = 0.5

    maxJ = np.max(np.abs(coupling))
    maxh = np.max(np.abs(model.h))
    max_attraction = maxJ + maxh
    frozen_nodes = np.argpartition(freeze_score, -nb_seeds)[-nb_seeds:]
    stack = frozen_nodes
    for frozen_node in stack:
        neighbours = np.where(coupling[frozen_node, :] != 0)[0]
        for neighbour in neighbours:
            neighbour_action = np.abs(coupling[frozen_node, neighbour] * sigma[frozen_node] + model.h[frozen_node])/max_attraction
            if neighbour_action >= threshold and neighbour not in frozen_nodes:
                frozen_nodes = np.append(frozen_nodes, neighbour)
                stack = np.append(stack, neighbour)
        stack = np.delete(stack, np.where(stack==frozen_node)[0])
    # frozen_nodes = np.random.choice(frozen_nodes, size=(final_size,))
    LOGGER.info(f"Amount of frozen nodes: {len(frozen_nodes)}")
    return frozen_nodes

def approximate_frozen_gradient_cluster(model:IsingModel, sigma:np.ndarray, nb_seeds:int, dt:float) ->np.ndarray:
    """
    Approximate the frozen gradient cluster by selecting the nodes with the highest gradient.
    """
    coupling = model.J + model.J.T
    gradient = -coupling @ sigma - model.h
    freeze_score = np.abs(gradient) / np.max(np.abs(gradient))

    frozen_nodes = np.argpartition(freeze_score, -nb_seeds)[-nb_seeds:]
    sigma[frozen_nodes] *= -1

    new_sigma, _ = Multiplicative().solve(model, sigma, dt, 5000, initial_temp_cont=0.0, frozen_nodes=frozen_nodes)
    LOGGER.info(np.where(new_sigma != sigma))
    frozen_nodes = np.block([frozen_nodes, np.where(new_sigma != sigma)[0]])
    LOGGER.info(f"Approximate frozen nodes: {len(frozen_nodes)}")
    return frozen_nodes