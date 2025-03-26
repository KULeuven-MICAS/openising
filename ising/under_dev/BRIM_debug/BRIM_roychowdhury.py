import numpy as np
import time
import pathlib

from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.flow import return_G

def BRIMroychowdhury(model: IsingModel,
        initial_state:  np.ndarray,
        num_iterations: int,
        dtBRIM:         float,
        C:              float,
        file:           pathlib.Path|None,
        stop_criterion: float = 1e-8,
        seed:           int   = 0,):
    
    N      = model.num_variables
    tend   = dtBRIM * num_iterations
    t_eval = np.linspace(0.0, tend, num_iterations)

    # Transform the model to one with no h and mean variance of J
    new_model = model.transform_to_no_h()
    J         = triu_to_symm(new_model.J)

    G = 1.#return_G(J) * 0.5

    # Add the bias node and add noise to the initial voltages
    v = np.block([0.5*initial_state, 1.0])
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)
    v += 0.001 * (np.random.random((N + 1,)) - 0.5)

    schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

    def dvdt(t, vt, coupling):
            k     = gain(t, tend)
            V_mat = np.array([vt] * vt.shape[0])
            dv    = 1/C * (G*(np.tanh(k*np.tanh(k*vt)) - vt) -np.sum(coupling*(V_mat.T-V_mat), axis=1))
            
            cond1 = (dv > 0) & (vt > 0)
            cond2 = (dv < 0) & (vt < 0)
            dv   *= np.where(cond1 | cond2, 1 - vt**2, 1)
            return dv
    
    with HDF5Logger(file, schema) as log:
        log_metadata(
            logger         = log,
            initial_state  = np.sign(v),
            model          = model,
            num_iterations = num_iterations,
            C              = C,
            time_step      = dtBRIM,
            seed           = seed,
            stop_criterion = stop_criterion,
        )

        i                 = 0
        previous_voltages = np.copy(v)
        max_change        = np.inf

        sample = np.sign(v[:N])
        energy = model.evaluate(sample)
        log.log(time_clock=0., energy=energy, state=sample, voltages=v[:N])
        while i < (num_iterations) and max_change > stop_criterion:
            tk = t_eval[i]

            # Runge Kutta steps
            k1 = dtBRIM * dvdt(tk, previous_voltages, J)
            k2 = dtBRIM * dvdt(tk + 2 / 3 * dtBRIM, previous_voltages + 2 / 3 * k1, J)

            new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2)

            max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                previous_voltages, ord=np.inf
            )

            # Log everything
            sample = np.sign(new_voltages[:N])*np.sign(new_voltages[-1])
            energy = model.evaluate(sample)
            log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            # Update criterion changes
            previous_voltages = np.copy(new_voltages)
            i += 1

        # Make sure to log to the last iteration if the stop criterion is reached
        if max_change < stop_criterion:
            for j in range(i, num_iterations):
                tk = t_eval[j]
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

        log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
    return sample, energy


def log_metadata(logger:HDF5Logger, model, initial_state, num_iterations, **kwargs):
    metadata = {
            "solver":"BRIM_roychowdhury",
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
    logger.write_metadata(**metadata)

def gain(t, tend):
    return 20/tend*t