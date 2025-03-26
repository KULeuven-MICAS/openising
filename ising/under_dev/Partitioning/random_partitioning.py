import numpy as np

from ising.model.ising import IsingModel

def random_partitioning(model:IsingModel):
    return np.random.choice([-1,1], (model.num_variables,))
