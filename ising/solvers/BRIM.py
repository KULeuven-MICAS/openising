import time
import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class BRIM(SolverBase):
    def __init__(self):
        self.name = "BRIM"

    def k(self, kmax: float, kmin: float, t: float, t_final: float) -> float:
        """Returns the gain of the latches at time t.

        Args:
            kmax (float): maximum gain of the latch
            kmin (float): minimum gain of the latch
            tend (float): end time of the simulation
            t (float): time

        Returns:
            float: latch gain
        """
        return kmin + ((kmax - kmin) / t_final) * t

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtBRIM: float,
        C: float,
        stop_criterion: float = 1e-8,
        file: pathlib.Path | None = None,
        initial_temp: float = 1.0,
        end_temp: float = 0.05,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Simulates the BLIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            initial_state (np.ndarray): initial spins of the nodes
            num_iterations (int): amount of iterations that need to be simulated
            dtBRIM (float): time step.
            C (float): capacitor parameter.
            stop_criterion (float, optional): stop criterion for the maximum allowed change between iterations.
                                              Defaults to 1e-8.
            file (pathlib.Path, None, Optional): absolute path to which data will be logged. If 'None',
                                                 nothing is logged.
            initial_temp (float, optional): initial temperature. Defaults to 1.0.
            end_temp (float, optional): end temperature. Defaults to 0.05.
            seed (int, optional): seed for the random number generator. Defaults to 0.

        Returns:
            sample,energy (tuple[np.ndarray, float]): the final state and energy of the system.
        """

        # Set the time evaluations
        tend   = dtBRIM * num_iterations
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
        v = np.block([0.01*initial_state, 1.0])
        v += 0.001 * (np.random.random((N + 1,)) - 0.5)

        # Schema for the logging
        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        def dvdt(t, vt, coupling):
            # Make sure the bias node is 1
            vt[-1] = 1.0

            # Compute the differential equation
            V_mat = np.array([vt] * vt.shape[0])
            dv = -1 / C * np.sum(coupling * (V_mat.T - V_mat), axis=1)

            # Make sure the voltages stay in the range [-1, 1]
            cond1 = (dv > 0) & (vt > 0)
            cond2 = (dv < 0) & (vt < 0)
            dv *= np.where(cond1 | cond2, 1 - vt**2, 1)

            # Make sure the bias node does not change
            dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            # Log the initial metadata
            self.log_metadata(
                logger         = log,
                initial_state  = np.sign(v),
                model          = model,
                num_iterations = num_iterations,
                C              = C,
                time_step      = dtBRIM,
                seed           = seed,
                temperature    = initial_temp,
                stop_criterion = stop_criterion,
            )

            # Initialize the simulation variables
            i                 = 0
            previous_voltages = np.copy(v)
            max_change        = np.inf
            T                 = initial_temp if initial_temp <= 1.0 else 1.0
            cooling_rate      = (end_temp / initial_temp) ** (1 / (num_iterations - 1)) if initial_temp != 0. else 1.

            # Initial logging
            sample = np.sign(v[:N])
            energy = model.evaluate(sample)
            log.log(time_clock=0., energy=energy, state=sample, voltages=v[:N])

            while i < (num_iterations) and max_change > stop_criterion:
                tk = t_eval[i]

                # Runge Kutta steps
                k1 = dtBRIM * dvdt(tk, previous_voltages, J)
                k2 = dtBRIM * dvdt(tk + 2 / 3 * dtBRIM, previous_voltages + 2 / 3 * k1, J)

                # Add noise and update the voltages
                noise = T * (np.random.random((N+1,)) - 0.5)
                cond1 = (previous_voltages > 1) & (noise > 0)
                cond2 = (previous_voltages < -1) & (noise < 0)
                noise *= np.where(cond1|cond2, 1-previous_voltages**2, 1)
                new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2) + noise

                # Lower the temperature
                T *= cooling_rate

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

            # Make sure to log to the last iteration if the stop criterion is reached
            if max_change < stop_criterion:
                for j in range(i, num_iterations):
                    tk = t_eval[j]
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy
