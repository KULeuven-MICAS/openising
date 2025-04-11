import numpy as np

from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm

def apply_partitioning(model: IsingModel, partitioning: np.ndarray) -> tuple[dict[int, IsingModel], dict[int, np.ndarray], dict[int, set]]:
    unique_partitions = np.unique(partitioning)
    nodes_partitions = {i: [] for i in unique_partitions}

    # Separate the nodes based on the partitioning
    for node, part in enumerate(partitioning):
        nodes_partitions[part].append(node)
    
    # Replicate nodes in order to have fully separated models
    replica_nodes = {i: set() for i in unique_partitions}
    tot_replica_nodes = 0
    triu_J = triu_to_symm(model.J)
    for i in range(len(unique_partitions)):
        part_id_i = unique_partitions[i]
        part_i = nodes_partitions[unique_partitions[i]]
        for j in range(i + 1, len(unique_partitions)):
            part_id_j = unique_partitions[j]
            part_j = nodes_partitions[unique_partitions[j]]

            for node_i in part_i:
                connected_nodes = np.nonzero(triu_J[node_i, :])[0]
                for other_node in connected_nodes:
                    if other_node in part_j:
                        replica_nodes[part_id_j].add(node_i)
                        replica_nodes[part_id_i].add(other_node)
    
        tot_replica_nodes += len(replica_nodes[part_id_i])

    # Create models for each partition 
    models = dict()
    node_maps = dict()
    constraints = dict()

    constr = 0
    for partition_id in unique_partitions:
        partition_nodes = list(set(nodes_partitions[partition_id]) | replica_nodes[partition_id])
        partition_nodes.sort()

        n = len(partition_nodes)
        J = np.zeros((n, n))
        h = np.zeros((n,))

        node_map = {node: idx for idx, node in enumerate(partition_nodes)}
        node_maps[partition_id] = node_map
        
        # Fill in J and h for this partition
        for i, node_i in enumerate(partition_nodes):
            h[i] = model.h[node_i]
            for j, node_j in enumerate(partition_nodes):
                J[i, j] = model.J[node_i, node_j]
        
        models[partition_id] = IsingModel(J, h)
        constraints[partition_id] = np.zeros((tot_replica_nodes, n))
    
    for i in range(len(unique_partitions)):
        partition_id = unique_partitions[i]
        for node, idx1 in node_maps[partition_id].items():
            if constr < tot_replica_nodes:
                for j in range(i, len(unique_partitions)):    # look in which other partition the replica node is stored
                    other_partition_id = unique_partitions[j]
                    if other_partition_id != partition_id and node_maps[other_partition_id].get(node) is not None:
                        # The partition should be different and the node should also be replicated
                        idx2 = node_maps[other_partition_id][node]
                        constraints[partition_id][constr, idx1] = 1
                        constraints[other_partition_id][constr, idx2] = -1
                        
                        # make sure the constraints have their own row
                        constr += 1

                        # Break to do this only once and go to the next constraint
                        break

    return models, constraints, replica_nodes