import numpy as np

from ising.flow import TOP, LOGGER
from ising.under_dev import MaxCutParser, TSPParser
from ising.generators.TSP import TSP
from ising.stages.model.ising import IsingModel

from ising.under_dev.Partitioning.modularity import partitioning_modularity
from ising.under_dev.Partitioning.SPLIT import SPLIT
from ising.under_dev.Partitioning.dual_decomposition import dual_decomposition
from ising.under_dev.Partitioning.apply_partitioning import apply_partitioning

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
    burma14, best_found = TSPParser.TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model = TSP(burma14, 1.2)

    nb_partitions = 4
    partitions, _ = partitioning_modularity(model, nb_partitions)

    sigma_init = np.random.choice([-1, 1], size=(model.num_variables,))
    hyperparameters = {"num_iter":50000, "nb_flipping": 100, "cluster_threshold":0.3, "init_cluster_size": 0.95, "end_cluster_size":0.02}
    num_iterations = 100

    # _, energy = SPLIT(partitions,  sigma_init, model, num_iterations,**hyperparameters)
    # LOGGER.info(f"SPLIT: Obtained energy: {energy:.2f}, Best found: {best_found:.2f}, relative error: {np.abs((energy - best_found) / best_found):.2%}")

    models, constraints, _ = apply_partitioning(model, partitions)
    init_states = {i: sigma_init[np.where(partitions==i)] for i in range(-int(nb_partitions/2), int(nb_partitions/2)+1)}
    init_states.pop(0)
    dual_decomposition(models, constraints, init_states, num_iterations, 0.1, **hyperparameters )


if __name__ == "__main__":
    test_SPLIT()