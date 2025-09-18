import numpy as np
from scipy.stats import norm


def calc_prob(size, noise_std):
    # calculate the probability of the ground state succeed to be the ground state for a specific GS-ES pair
    # the prob is the CDF of a standard norm distribution at point z
    # parameters:
    delta_E = 20
    W = size
    D = size
    B = 1  # facotr of deciding ES count
    k = 1  # factor of deciding ES count
    N_ES = np.exp(B * (size**k))
    N_GS = 2
    z = delta_E / (2 * noise_std * np.sqrt(W + D))
    prob_fail_single_pair = norm.cdf(-z)
    prob_success_single_gs = (1 - prob_fail_single_pair) ** N_ES
    prob_success = 1 - (1 - prob_success_single_gs) ** N_GS
    prob_fail = 1 - prob_success
    print(z, prob_fail_single_pair, prob_success_single_gs, prob_fail)
    return prob_success


if __name__ == "__main__":
    # derive the probability of the ground state succeed to be the ground state in Ising
    size_prob = 100
    noise_std = 0.05
    prob_success = calc_prob(size_prob, noise_std)
    print(prob_success)
