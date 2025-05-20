import numpy as np

from ising.model.ising import IsingModel


def knapsack(profit: np.ndarray, capacity: int, weights: np.ndarray, penalty_value: float) -> IsingModel:
    """Generates an instance of the knapsack problem in the Ising form.

    Args:
        profit (np.ndarray): the profits of choosing the items.
        capacity (int): the capacity of the knapsack.
        weights (np.ndarray): the weight of every item.
        penalty_value (float): the penalty value for the constraint.

    Returns:
        IsingModel: the corresponding Ising model.
    """
    alpha = np.max(profit) * penalty_value

    N = len(weights)
    nb_bits = int(np.floor(np.log2(capacity) + 1))

    coupling = np.zeros((N + nb_bits, N + nb_bits))
    h = np.zeros((N + nb_bits,))
    constant = 0

    # Add profit terms
    coupling[:N, :N] = profit/4
    np.fill_diagonal(coupling, 0)

    for i in range(N):
        h[i] += profit[i, i] / 2

    for i in range(N):
        profit_sum = 0
        for j in range(N):
            if i != j:
                profit_sum += profit[i,j]
        h[i] += 1/4*profit_sum

    for i in range(N):
        for j in range(N):
            if i == j:
                constant -= profit[i, j] / 2
            else:
                constant -= profit[i,j] / 8

    # Add weight terms
    weight_sum = np.sum(weights)
    for i in range(N):
        weight_i = weights[i]
        for j in range(N):
            weight_j = weights[j]
            if i != j:
                coupling[i, j] -= 1/2*weight_i*weight_j*alpha
            else:
                constant += 1/4*alpha*weight_i*weight_j
            constant += 1/4*alpha*weights[i] * weights[j]
        h[i] -= 1/2*weight_i*(weight_sum)*alpha

    for i in range(N):
        h[i] += capacity*weights[i]*alpha
        constant -= alpha*capacity*weights[i]

    # Add slack variable
    slack_sum = np.sum([2**q for q in range(nb_bits)])
    for k in range(nb_bits):
        for q in range(nb_bits):
            if k != q:
                coupling[N+k, N+q] -= 1/2*(2**(q+k))*alpha
            else:
                constant += 1/4*(2**(q+k))*alpha
            constant += 1/4*alpha*(2**(q+k))
        h[N+k] -= 1/2*(2**k)*(slack_sum)*alpha

    for q in range(nb_bits):
        h[N+q] += capacity*alpha*(2**q)
        constant -= capacity *alpha*(2**q)

    # Add weight-slack terms
    for i in range(N):
        weight_i = weights[i]
        for q in range(nb_bits):
            coupling[i, N+q] -= 1/2*weight_i*(2**q)*alpha
            coupling[N+q, i] -= 1/2*weight_i*(2**q)*alpha
        h[i] -= 1/2*weight_i*slack_sum*alpha

    for q in range(nb_bits):
        h[N+q] -= 1/2*weight_sum*(2**q)*alpha

    constant += 1/2*alpha*weight_sum*slack_sum

    # Add slack variables constraint
    # for q in range(nb_bits):
    #     for k in range(nb_bits):
    #         if q != k:
    #             coupling[N+q, N+k] -= 1/2*(2**(q+k))*alpha
    #         else:
    #             constant += 1/4*(2**(q+k))*alpha
    #         constant += 1/4*(2**(q+k))*alpha
    #     h[N+q] -= 1/2*(2**q)*slack_sum*alpha

    constant += alpha*(capacity**2)

    coupling = np.triu(coupling, 1)
    model = IsingModel(coupling, h, constant, name="Knapsack")

    return model
