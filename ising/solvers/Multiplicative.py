import numpy as np
import pathlib
import time

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.flow import return_rx


class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        seed:int = 0,
        initial_temp:float = 1.,
        end_temp: float = 0.05,
        stop_criterion: float = 1e-8,
        file: pathlib.Path|None=None,
    ) -> tuple[float, np.ndarray]:
        """Solves the given problem using a multiplicative coupling scheme.

        Args:
            model (IsingModel): the model to solve.
            initial_state (np.ndarray): the initial spins of the nodes.
            dtMult (float): time step.
            num_iterations (int): the number of iterations.
            file (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: the best energy and the best sample.
        """
        # Set up the time evaluations
        tend   = dtMult * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations)

        # Transform the model to one with no h and mean variance of J
        model.normalize()
        new_model = model.transform_to_no_h()
        J = triu_to_symm(new_model.J)
        model.reconstruct()

        # make sure the correct random seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the bias node and add noise to the initial voltages
        N = model.num_variables
        v = np.block([0.1*initial_state, 1.0])
        v += 0.01 * (np.random.random((N+1,)) - 0.5)

        # Schema for logging
        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        # Define the system equations
        def dvdt(t, vt, coupling):
            # Set the bias node to 1.
            vt[-1] = 1.0

            # Compute the voltage change dv
            k = np.tanh(3*vt)
            dv = 1 / 2 * np.dot(coupling, k)

            # Ensure the voltages stay in the range [-1, 1]
            cond1 = (dv > 0) & (vt > 0)
            cond2 = (dv < 0) & (vt < 0)
            dv *= np.where(cond1 | cond2, 1 - v**2, 1)

            # Ensure the bias node does not change
            dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger         = log,
                initial_state  = np.sign(v[:-1]),
                model          = model,
                num_iterations = num_iterations,
                time_step      = dtMult,
                temperature   = initial_temp,
            )

            # Set up the simulation
            i                 = 0
            max_change        = np.inf
            previous_voltages = np.copy(v)
            T                 = initial_temp if initial_temp < 1.0 else 1.0
            cooling_rate      = return_rx(num_iterations, initial_temp, end_temp)

            while i < num_iterations and max_change > stop_criterion:
                tk = t_eval[i]

                # Runge Kutta steps
                k1 = dtMult * dvdt(tk, v, J)
                k2 = dtMult * dvdt(tk + 2 / 3 * dtMult, v + 2 / 3 * k1, J)
                new_voltages = previous_voltages +  1.0 / 4.0 * (k1 + 3.0 * k2) + T * (np.random.random((N+1,)) - 0.5)

                T *= cooling_rate

                # Log everything
                sample = np.sign(new_voltages[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

                # Update the criterion changes
                max_change        = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                                        previous_voltages, ord=np.inf
                                    )
                previous_voltages = np.copy(new_voltages)
                i                += 1

            # Make sure to log to the last iteration if the stop criterion is reached
            if max_change < stop_criterion:
                for j in range(i, num_iterations):
                    tk = t_eval[j]
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy
