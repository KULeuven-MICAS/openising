import numpy as np

from ising.flow import TOP, LOGGER
from ising.under_dev import MaxCutParser
from ising.stages.model.ising import IsingModel

from ising.under_dev.Partitioning.modularity import partitioning_modularity
from ising.under_dev.Partitioning.SPLIT import SPLIT

from ising.utils.flow import return_c0

figtop = TOP / "ising/under_dev/Partitioning/figures"

def optimal_state_from_partitioning(optimal_states:dict[int: np.ndarray], model: IsingModel, partitioning: np.ndarray, replica_nodes: dict[int:np.ndarray]):
    state = np.zeros((model.num_variables,))
    partitions = np.unique(partitioning)
    
    nodes_partitions = {i:[] for i in np.unique(partitioning)}
    node_maps = dict()
    for node, part in enumerate(partitioning):
        nodes_partitions[part].append(node)

    for _, part in enumerate(partitions):
        part_nodes = set(nodes_partitions[part])
        part_nodes = list(part_nodes | replica_nodes[part])
        part_nodes.sort()

        node_map = {node: idx for idx, node in enumerate(part_nodes)}
        node_maps[part] = node_map


    for node, part in enumerate(partitioning):
        amount_replicas = 3
        avg_node = 0
        for other_part, replica_node in replica_nodes.items():
            if node in replica_node and other_part != part:
                amount_replicas += 1
                avg_node += optimal_states[other_part][node_maps[other_part][node]]
        avg_node += optimal_states[part][node_maps[part][node]]*3
        avg_node /= amount_replicas
        if avg_node == 0:
            state[node] = optimal_states[part][node_maps[part][node]]
        else:
            state[node] = np.sign(avg_node)

    energy = model.evaluate(state)

    return state, energy

def test_SPLIT():
    g16, best_found = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/G16.txt")
    model = MaxCutParser.generate_maxcut(g16)

    nb_partitions = 4
    partitions, _ = partitioning_modularity(model, nb_partitions)

    sigma_init = np.random.choice([-1, 1], size=(model.num_variables,))
    solver = "bSB"
    c0 = return_c0(model)
    hyperparameters = {"num_iter": 2000, "a0":1.0, "c0": c0, "dtSB":1.0}
    num_iterations = 100

    optimal_states, energy = SPLIT(model, partitions, num_iterations, sigma_init, solver, **hyperparameters)
    LOGGER.info(f"Obtained energy: {energy:.2f}, Best found: {best_found:.2f}, relative error: {(energy - best_found) / best_found:.2%}")
    LOGGER.info(f"Optimal state: {optimal_states}")

if __name__ == "__main__":
    test_SPLIT()