import numpy as np
import time
import pathlib

from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger

def Multiplicative(model: IsingModel,
        initial_state:  np.ndarray,
        num_iterations: int,
        dtBRIM:         float,
        C:              float,
        file:           pathlib.Path|None,
        random_flip:    bool  = False,
        initial_temp:   float = 50.0,
        cooling_rate:   float = 0.9,
        seed:           int   = 0,
        stop_criterion: float = 1e-8):
    
    # Set up time evaluations
    tend   = dtBRIM * num_iterations
    t_eval = np.linspace(0.0, tend, num_iterations)

    # Transform the model to one with no h and mean variance of J
    model.normalize()
    new_model = model.transform_to_no_h()
    J = triu_to_symm(new_model.J)
    model.reconstruct()

    # Make sure the correct random seed is used
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)
    
    # Add the bias node and add noise to the initial voltages
    N  = model.num_variables
    v  = np.block([0.1*initial_state, 1.0])
    v += 0.01 * (np.random.random((N + 1,)) - 0.5)

    # Schema for the logging
    schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

    # Define the system equations
    def dvdt(t:float, vt:np.ndarray, coupling:np.ndarray):
        """Differential equations for the multiplicative BRIM model.

        Args:
            t (float): time
            vt (np.ndarray): current voltages
            coupling (np.ndarray): coupling matrix J

        Returns:
            _type_: _description_
        """
        # set bias node to 1.
        vt[-1] = 1.0

        # vt[np.where(np.abs(vt) > 1)] = np.sign(vt[np.where(np.abs(vt) > 1)])

        # Compute the voltage change dv
        k  = np.tanh(3*vt)
        dv = 1 / 2 * np.dot(coupling, k)

        # Ensure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 0)
        cond2 = (dv < 0) & (vt < 0)
        dv   *= np.where(cond1|cond2, 1-vt**2, 1)

        # Ensure the bias node does not change
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

        i                 = 0
        max_change        = np.inf
        previous_voltages = np.copy(v)
        T                 = initial_temp

        sample = np.sign(v[:N])
        energy = model.evaluate(sample)
        log.log(time_clock=0., energy=energy, state=sample, voltages=v[:N])
        
        while i < num_iterations and max_change > stop_criterion:
            tk = t_eval[i]

            # Perform the random flip annealing wise
            if random_flip:
                rand = np.random.random()
                if rand < np.exp(-1 / T):
                    flip                    = np.random.choice(N)
                    previous_voltages[flip] = -previous_voltages[flip]
            T *= cooling_rate

            # Runge Kutta steps
            k1 = dtBRIM * dvdt(tk, previous_voltages, J)
            k2 = dtBRIM * dvdt(tk + 2 / 3 * dtBRIM, previous_voltages + 2 / 3 * k1, J)

            new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2)

            # Log everything
            sample = np.sign(new_voltages[:N])*np.sign(new_voltages[-1])
            energy = model.evaluate(sample)
            log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            # Update the criterion changes
            max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                previous_voltages, ord=np.inf
            )
            previous_voltages = np.copy(new_voltages)
            i += 1

        # Make sure to log to the last iteration if the stop criterion is reached
        if max_change < stop_criterion:
            for j in range(i, num_iterations):
                tk = t_eval[j]
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

        # write final metadata
        log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
    return sample, energy


def log_metadata(logger:HDF5Logger, model, initial_state, num_iterations, **kwargs):
    metadata = {
            "solver":"Multiplicative",
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
    logger.write_metadata(**metadata)