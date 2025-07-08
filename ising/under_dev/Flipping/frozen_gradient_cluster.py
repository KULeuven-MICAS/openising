import numpy as np

from ising.stages.model.ising import IsingModel
from ising.flow import LOGGER

def find_frozen_gradient_cluster(model:IsingModel, sigma:np.ndarray, threshold:float, nb_seeds:int, final_size:int) ->np.ndarray:
    coupling = model.J + model.J.T
    gradient = -coupling @ sigma - model.h
    freeze_score = np.abs(gradient) / np.max(np.abs(gradient))

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
    frozen_nodes = np.random.choice(frozen_nodes, size=(final_size,))
    # LOGGER.info(f"Amount of frozen nodes: {len(frozen_nodes)}")
    return frozen_nodes