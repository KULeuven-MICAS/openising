import numpy as np

from ising.model.ising import IsingModel
from ising.utils.flow import run_solver
from ising.utils.numpy import triu_to_symm

def dual_decomposition(models: dict[int: IsingModel], constraints: dict[int: np.ndarray], initial_states:dict[int:  np.ndarray],
                      num_iterations:int, solver: str, step:float, stop_criterion: float = 1e-8, **hyperparameters) ->tuple[dict[int: np.ndarray], np.ndarray]:
    
    partitions = list(models.keys())
    lambda_k = np.zeros((constraints[partitions[0]].shape[0],))

    k = 0
    max_change = np.inf
    lambda_new = np.zeros_like(lambda_k)
    while k < num_iterations and max_change > stop_criterion:
        for _, partition in enumerate(partitions):
            lam = lambda_k.T @ constraints[partition]

            models[partition].h += lam
            initial_states[partition], _ = run_solver(solver, num_iterations*10, initial_states[partition], models[partition], **hyperparameters)
            models[partition].h -= lam
            lambda_new += step * constraints[partition] @ initial_states[partition] 

        # Update the dual variable
        lambda_new += lambda_k 

        k += 1
        max_change = np.linalg.norm(lambda_new - lambda_k, ord=np.inf) / (np.linalg.norm(lambda_k, ord=np.inf) if np.linalg.norm(lambda_k, ord=np.inf) > 0 else 1)
        lambda_k = lambda_new
    
    return initial_states, lambda_k
