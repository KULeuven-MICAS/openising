import numpy as np

from ising.stages.model import IsingModel

def compute_ruggedness(model: IsingModel, nb_samples:int, distance: int= 1) -> float:
    """
    """
    all_energies = []
    random_walk = generate_random_walk(model.num_variables, nb_samples)
    random_walk_avg = 0
    for ind in range(nb_samples):
        all_energies.append(model.evaluate(random_walk[ind, :].astype(np.float32)))
    avg_energy = np.mean(all_energies)
    for ind in range(nb_samples - distance):
        random_walk_avg += (all_energies[ind] - avg_energy)*(all_energies[ind+distance] - avg_energy)
    random_walk_avg /= (nb_samples - distance)
    ruggedness = random_walk_avg / np.var(all_energies)
    return ruggedness

def generate_random_walk(num_variables: int, nb_samples:int) -> np.ndarray:
    states = np.zeros((nb_samples, num_variables), dtype=int)
    states[0, :] = np.random.choice([-1, 1], size=num_variables)
    memory = []
    for i in range(1, nb_samples):
        allowed = np.setdiff1d(np.arange(num_variables), memory)
        flip_index = np.random.choice(allowed)
        states[i, :] = states[i-1, :]
        states[i, flip_index] *= -1
        memory.append(flip_index)
        if len(memory) >= num_variables/10:
            memory.pop(0)
    return states
