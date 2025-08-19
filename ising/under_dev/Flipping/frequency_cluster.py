import numpy as np

def frequency_cluster(count:np.ndarray, cluster_size:int, threshold:float) -> np.ndarray:
    freq = count / (np.max(np.abs(count)) if np.max(np.abs(count)) != 0 else 1)
    available_nodes = np.where(freq < threshold)[0]
    current_size = len(available_nodes)
    if len(available_nodes) <= cluster_size:
        ind_unavailable_nodes = np.where(freq >= threshold)[0]
        chosen_nodes = np.array([], dtype=int)
        while len(chosen_nodes) < cluster_size - current_size:
            chosen_nodes = np.unique(np.append(chosen_nodes, np.random.choice(ind_unavailable_nodes, (cluster_size - current_size - len(chosen_nodes),))))
        available_nodes = np.append(available_nodes, chosen_nodes)
        cluster = available_nodes
    else:
        cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
    return cluster