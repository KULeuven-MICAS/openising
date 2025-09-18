import numpy as np

def smoothening_cluster(points:list[tuple[np.ndarray,float]], max_size:int, choice:str="smallest")->np.ndarray:
    weight_nodes = np.zeros_like(points[0][0], dtype=float)
    for point, en in points:
        weight_nodes += 1/en * point # the smaller the energy, the larger the weight
    if choice == "smallest":
        cluster = np.argsort(weight_nodes)[:max_size]
    else:
        cluster = np.argsort(weight_nodes)[-max_size:]

    return cluster