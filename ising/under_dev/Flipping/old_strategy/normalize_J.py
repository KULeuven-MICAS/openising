import numpy as np

from ising.model.ising import IsingModel

def normalize_J(model:IsingModel, other_model:IsingModel) -> tuple[float, float, float]:
    # transform the model to have no h
    other_model = other_model.transform_to_no_h()
    N = model.num_variables
    model = model.transform_to_no_h()
   
    other_J = other_model.J[:N, :N]
    other_h = np.reshape(other_model.J[:N, -1], (-1, 1))
    other_J = np.block([[other_J,other_h], [other_h.T, 0]])

    mean_J = np.mean(model.J)
    mean_other_J = np.mean(other_J)
    print(mean_other_J)

    cov = np.mean((model.J - mean_J)*(other_J - mean_other_J))
    var = np.var(model.J - mean_J)

    a = cov / var
    b = mean_other_J - a * mean_J

    model.J = a*model.J + b
    rmse = np.sqrt(np.mean((model.J - other_J) ** 2))

    return model, rmse, a, b
