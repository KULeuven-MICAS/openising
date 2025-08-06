import numpy as np

from ising.flow import LOGGER
from ising.stages.model.ising import IsingModel
# from ising.under_dev import sim_stage
from ising.under_dev.Flipping.new_strategy import do_flipping

def SPLIT(partitions:list[int], sigma_init:np.ndarray, model: IsingModel,  num_iterations:int, **hyperparameters):
    partitioned_models, nodes_per_partition = partition_model(model, partitions)

    sigma = sigma_init.copy()
    energy_old = np.inf
    for i in range(num_iterations):
        local_fields = compute_local_fields(model, sigma, nodes_per_partition)
        for part_id, part_model in partitioned_models.items():
            part_model.h += local_fields[part_id]
            _,  energy, sigma[nodes_per_partition[part_id]] = do_flipping(cluster_size_init=int(hyperparameters["cluster_size_init"]*part_model.num_variables),
                                                                      cluster_size_end=int(hyperparameters["cluster_size_end"]*part_model.num_variables),
                                                                      sigma_init=sigma[nodes_per_partition[part_id]],
                                                                      model=part_model,
                                                                      cluster_threshold=hyperparameters["threshold"],
                                                                      nb_flipping=hyperparameters["nb_flipping"],
                                                                      dt=hyperparameters["dt"],
                                                                      num_iterations=hyperparameters["num_iter"],
                                                                      )
            part_model.h -= local_fields[part_id]
        
        energy_old = energy
        energy = model.evaluate(sigma)
        if energy == energy_old:
            return sigma, energy
        sigma = sweep_update(sigma, model)
    energy = model.evaluate(sigma)
    LOGGER.info(f"amount of cores: {len(np.unique(partitions))} - final energy: {energy}")
    return sigma, energy

def partition_model(model: IsingModel, partitions:list[int])->tuple[dict[int:IsingModel], dict[int:np.ndarray]]:
    part_ids = np.unique(partitions)
    models = {part_id: None for part_id in part_ids}
    nodes = np.arange(model.num_variables)
    nodes_per_partition = {part_id: nodes[partitions == part_id] for part_id in part_ids}
    for part_id in part_ids:
        models[part_id] = IsingModel(model.J[nodes_per_partition[part_id],:][:, nodes_per_partition[part_id]],
                                     model.h[nodes_per_partition[part_id]])
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
