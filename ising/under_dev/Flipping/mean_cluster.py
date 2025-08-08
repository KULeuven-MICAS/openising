import numpy as np

def find_cluster_mean(best_sigmas, sigma, max_size, threshold, option):
    mean_sigma = np.mean(best_sigmas, axis=0)

    if option == 1:
        available_nodes = np.where(np.abs(mean_sigma) >= threshold, np.arange(len(sigma)), -1)
    else:
        available_nodes = np.where(np.abs(mean_sigma) <= threshold, np.arange(len(sigma)), -1)
   
    if len(available_nodes[available_nodes >= 0]) < max_size:
        current_size = len(available_nodes[available_nodes >= 0])
        ind_unavailable_nodes = np.where(available_nodes < 0)[0]
        chosen_nodes = np.random.choice(ind_unavailable_nodes, (max_size - current_size,))
        available_nodes[chosen_nodes] = np.arange(len(available_nodes))[chosen_nodes]


    cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(max_size,))
    return cluster