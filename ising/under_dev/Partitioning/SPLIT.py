import numpy as np

from ising.flow import LOGGER
from ising.stages.model.ising import IsingModel
# from ising.under_dev import sim_stage
from ising.solvers.Multiplicative import Multiplicative

def SPLIT(partitions:list[int], sigma_init:np.ndarray, model: IsingModel,  num_iterations:int, **hyperparameters):
    partitioned_models, nodes_per_partition = partition_model(model, partitions)

    sigma = sigma_init.copy()
    energy_old = np.inf
    all_times = {part_id: 0.0 for part_id in partitioned_models.keys()} 
    for i in range(num_iterations):
        local_fields = compute_local_fields(model, sigma, nodes_per_partition)
        for part_id, part_model in partitioned_models.items():
            part_model.h += local_fields[part_id]
            part_sigma, _, time = Multiplicative().solve(
                            model=part_model,
                            init_cluster_size=hyperparameters["cluster_size_init"],
                            end_cluster_size=hyperparameters["cluster_size_end"],
                            initial_state=sigma[nodes_per_partition[part_id]],
                            cluster_threshold=hyperparameters["threshold"],
                            nb_flipping=hyperparameters["nb_flipping"],
                            dtMult=hyperparameters["dt"],
                            num_iterations=hyperparameters["num_iter"],
                            initial_temp_cont=0.0)
            part_model.h -= local_fields[part_id]
            all_times[part_id] += time
            sigma[nodes_per_partition[part_id]] = part_sigma
        energy = model.evaluate(sigma)
        if energy == energy_old:
            LOGGER.info(f"amount of cores: {len(np.unique(partitions))} - final energy: {energy}")
            return sigma, energy
        energy_old = energy
        sigma = sweep_update(sigma, model)
    energy = model.evaluate(sigma)
    LOGGER.info(f"amount of cores: {len(np.unique(partitions))} - final energy: {energy}")
    return sigma, energy, max(all_times.values())

def partition_model(model: IsingModel, partitions:list[int])->tuple[dict[int:IsingModel], dict[int:np.ndarray]]:
    part_ids = np.unique(partitions)
    models = {part_id: None for part_id in part_ids}
    nodes = np.arange(model.num_variables)
    nodes_per_partition = {part_id: nodes[partitions == part_id] for part_id in part_ids}
    for part_id in part_ids:
        models[part_id] = IsingModel(np.copy(model.J[nodes_per_partition[part_id],:][:, nodes_per_partition[part_id]]),
                                     np.copy(model.h[nodes_per_partition[part_id]]))
    return models, nodes_per_partition

def compute_local_fields(model: IsingModel, sigma:np.ndarray, nodes_per_partition: dict[int:np.ndarray])->dict[int:np.ndarray]:
    local_fields = {part_id: np.zeros((len(nodes_per_partition[part_id]),)) for part_id in nodes_per_partition.keys()}
    coupling = model.J + model.J.T

    for part_id, nodes in nodes_per_partition.items():
        for other_part_id, other_nodes in nodes_per_partition.items():
            if part_id != other_part_id:
                local_fields[part_id] += coupling[nodes, :][:, other_nodes] @ sigma[other_nodes]
    return local_fields

def sweep_update(sigma: np.ndarray, model:IsingModel) -> np.ndarray:
    current_energy = model.evaluate(sigma)
    current_sigma = sigma.copy()
    for i in range(model.num_variables):
        current_sigma[i] *= -1
        new_energy = model.evaluate(current_sigma)
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            current_sigma[i] *= -1
    return current_sigma
