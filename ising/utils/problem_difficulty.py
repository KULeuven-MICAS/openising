import numpy as np

from ising.stages.model import IsingModel

def compute_ruggedness(model: IsingModel, nb_samples:int, distance: int= 1) -> float:
    """
    """
    all_energies = []
    random_walk = generate_random_walk(model.num_variables, nb_samples)
    random_walk_avg = 0
    for ind in range(nb_samples):
        if ind + distance < nb_samples:
            cost1 = model.evaluate(random_walk[ind, :].astype(np.float32))
            cost2 = model.evaluate(random_walk[ind + distance, :].astype(np.float32))
            random_walk_avg += (cost1 - cost2)**2
            all_energies.append(cost1)
        else:
            all_energies.append(model.evaluate(random_walk[ind, :].astype(np.float32)))
    random_walk_avg /= (nb_samples - distance)
    ruggedness = 1 - random_walk_avg / (2*(np.mean(np.power(all_energies, 2)) - np.mean(all_energies)**2))
    return ruggedness

def generate_random_walk(num_variables: int, nb_samples:int) -> np.ndarray:
    states = np.zeros((nb_samples, num_variables), dtype=int)
    states[0, :] = np.random.choice([-1, 1], size=num_variables)

    for i in range(1, nb_samples):
        flip_index = np.random.randint(0, num_variables)
        states[i, :] = states[i-1, :]
        states[i, flip_index] *= -1
    return states
