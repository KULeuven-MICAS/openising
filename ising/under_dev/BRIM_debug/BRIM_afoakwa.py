import numpy as np
import time
import pathlib

from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger

def BRIMafoakwa(model: IsingModel,
        initial_state:  np.ndarray,
        num_iterations: int,
        dtBRIM:         float,
        C:              float,
        file:           pathlib.Path|None,
        stop_criterion: float = 1e-8,
        initial_temp:   float = 50.0,
        cooling_rate:   float = 0.9,
        seed:           int = 0,):
    
    N      = model.num_variables
    tend   = dtBRIM * num_iterations
    t_eval = np.linspace(0.0, tend, num_iterations)

    T_J  = initial_temp*1.5
    r_TJ = (1. / T_J) ** (1/(int(0.8*num_iterations) + 1))

    # Transform the model to one with no h and mean variance of J
    new_model = model.transform_to_no_h()
    J = triu_to_symm(new_model.J)

    # Add the bias node
    v = np.block([0.1*initial_state, 1.0])

    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)
    v += 0.001 * (np.random.random((N + 1,)) - 0.5)

    schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

    def dvdt(t, vt, coupling):
            dv    = 1/C * (np.dot(coupling, vt) - gziv(vt)) 
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
            temperature    = initial_temp,
            cooling_rate   = cooling_rate,
            stop_criterion = stop_criterion,
        )

        i                 = 0
        previous_voltages = np.copy(v)
        max_change        = np.inf
        T                 = initial_temp

        sample = np.sign(v[:N])
        energy = model.evaluate(sample)
        log.log(time_clock=0., energy=energy, state=sample, voltages=v[:N])

        while i < (num_iterations) and max_change > stop_criterion:
            tk = t_eval[i]

            if i < int(0.8*num_iterations):
                coupling = J / T_J
            else:
                coupling = J

            # Runge Kutta steps
            k1 = dtBRIM * dvdt(tk, previous_voltages, coupling)
            k2 = dtBRIM * dvdt(tk + 2 / 3 * dtBRIM, previous_voltages + 2 / 3 * k1, coupling)

            new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2)
           
            # Do random flipping annealing wise
            rand = np.random.random()
            if rand < np.exp(-1/ T):
                flip               = np.random.choice(N)
                new_voltages[flip] = -new_voltages[flip]
            
            # Change annealing temperatures
            T   *= cooling_rate
            T_J *= r_TJ

            # Log everything
            sample = np.sign(new_voltages[:N])*np.sign(new_voltages[-1])
            energy = model.evaluate(sample)
            log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            # Update criterion changes
            max_change        = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                                previous_voltages, ord=np.inf
                                )   
            previous_voltages = np.copy(new_voltages)
            i                += 1

        # Make sure to log to the last iterations if the stop criterion is reached
        if max_change < stop_criterion:
            for j in range(i, num_iterations):
                tk = t_eval[j]
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

        log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
    return sample, energy


def log_metadata(logger:HDF5Logger, model, initial_state, num_iterations, **kwargs):
    metadata = {
            "solver":"BRIM_afoakwa",
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
    logger.write_metadata(**metadata)

def gziv(vt):
    large_1 = np.where(vt > 1)
    smal_1 = np.where(vt < -1)
    rest = np.where(np.logical_and(vt <= 1, vt >= -1))
    out = np.zeros_like(vt)
    out[large_1] = np.pi/2*(vt[large_1]-1)
    out[smal_1] = np.pi/2*(vt[smal_1]+1)
    out[rest] = 0.5*np.sin(-np.pi*vt[rest])

    return out