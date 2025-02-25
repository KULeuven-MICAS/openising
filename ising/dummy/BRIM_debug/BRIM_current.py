import numpy as np
import time
import pathlib

from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.flow import return_G

def BRIMcurrent(model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtBRIM: float,
        C: float,
        file:pathlib.Path|None,
        random_flip: bool = False,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.9,
        seed: int = 0,
        stop_criterion:float = 1e-8):
    # Set the time evaluations
    tend = dtBRIM * num_iterations
    t_eval = np.linspace(0.0, tend, num_iterations)

    # Transform the model to one with no h and mean variance of J
    model.normalize()
    new_model = model.transform_to_no_h()
    J = triu_to_symm(new_model.J)
    model.reconstruct()

    # Make sure the correct seed is used
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)

    # Ensure the bias node is added and add noise to the initial voltages
    N = model.num_variables
    v = np.block([0.5*initial_state, 1.0])
    v += 0.01 * (np.random.random((N + 1,)) - 0.5)

    # Schema for the logging
    schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

    # Define the system equations
    def dvdt(t, vt, coupling):
            # Make sure the bias node is 1
            vt[-1] = 1.0

            # Compute the differential equation
            V_mat = np.array([vt] * vt.shape[0])
            V_diff = V_mat.T - V_mat
            dv = -1 / C * np.sum(coupling * V_diff, axis=1) 

            # Make sure the voltages stay in the range [-1, 1]
            cond1 = (dv > 0) & (vt > 0)
            cond2 = (dv < 0) & (vt < 0)
            dv *= np.where(cond1 | cond2, 1 - vt**2, 1)

            # Make sure the bias node does not change
            dv[-1] = 0.0
            return dv
    
    with HDF5Logger(file, schema) as log:
        # Log initial metadata
        log_metadata(
            logger         = log,
            initial_state  = np.sign(v),
            model          = model,
            num_iterations = num_iterations,
            C              = C,
            time_step      = dtBRIM,
            random_flip    = random_flip,
            seed           = seed,
            temperature    = initial_temp,
            cooling_rate   = cooling_rate,
        )

        # Initialize simulation variables
        i                 = 0
        previous_voltages = np.copy(v)
        max_change        = np.inf
        T                 = initial_temp    

        # Initial logging
        sample = np.sign(v[:N])
        energy = model.evaluate(sample)
        log.log(time_clock=0., energy=energy, state=sample, voltages=v[:N])
        
        while i < (num_iterations) and max_change > stop_criterion:
            tk = t_eval[i]
            
            # Perform random flipping 
            if random_flip:
                rand = np.random.random()
                if rand < np.exp(-1 / T):
                    flip = np.random.choice(N)
                    previous_voltages[flip] = -previous_voltages[flip]
                T *= cooling_rate

            # Runge Kutta steps
            k1 = dtBRIM * dvdt(tk, previous_voltages, J)
            k2 = dtBRIM * dvdt(tk + 2 / 3 * dtBRIM, previous_voltages + 2 / 3 * k1, J)

            new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2)

            # Log everything
            sample = np.sign(new_voltages[:N])
            energy = model.evaluate(sample)
            log.log(time_clock=tk + dtBRIM, energy=energy, state=sample, voltages=new_voltages[:N])

            # Update criterion changes
            max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                previous_voltages, ord=np.inf
            )
            previous_voltages = np.copy(new_voltages)
            i += 1

        if max_change < stop_criterion:
            for j in range(i, num_iterations):
                tk = t_eval[j]
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

        log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
    return sample, energy


def log_metadata(logger:HDF5Logger, model, initial_state, num_iterations, **kwargs):
    metadata = {
            "solver":"BRIM_current",
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
    logger.write_metadata(**metadata)
