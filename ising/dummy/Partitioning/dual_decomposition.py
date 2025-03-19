import numpy as np

from ising.model.ising import IsingModel
from ising.utils.flow import run_solver
from ising.utils.numpy import triu_to_symm

def dual_decomposition(model1: tuple[np.ndarray, IsingModel], model2: tuple[np.ndarray, IsingModel],
                       A: np.ndarray, C:np.ndarray, num_iterations:int, solver: str, step:float, stop_criterion: float = 1e-8, **hyperparameters) -> tuple[np.ndarray, IsingModel]:
    s1 = model1[0]
    s2 = model2[0]

    m1 = model1[1]
    m2 = model2[1]

    lambda_k = np.ones((A.shape[0],))*np.max([np.average(np.sum(np.abs(triu_to_symm(m1.J)), axis=1)), np.average(np.sum(np.abs(triu_to_symm(m2.J)), axis=1))])

    k = 0
    max_change = np.inf
    while k < num_iterations and max_change > stop_criterion:

        lam1 = lambda_k.T @ A
        lam2 = lambda_k.T @ C

        # Solve the first model
        m1.h += lam1
        s1, _ = run_solver(solver, num_iterations*10, s1, m1, **hyperparameters)
        m1.h -= lam1

        # Solve the second model
        m2.h += lam2
        s2, _ = run_solver(solver, num_iterations*10, s2, m2, **hyperparameters)
        m2.h -= lam2

        # Update the dual variable
        lambda_new = lambda_k +  step * (A @ s1 + C @ s2)

        k += 1
        max_change = np.linalg.norm(lambda_new - lambda_k, ord=np.inf) / np.linalg.norm(lambda_k, ord=np.inf)
        lambda_k = lambda_new
    
    return s1, s2, lambda_k
