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
        v: np.ndarray,
        num_iterations: int,
        dt: float,
        C: float,
        stop_criterion: float = 1e-12,
        file: pathlib.Path | None = None,
        random_flip: bool = False,
        Temp: float = 50.0,
        r_T: float = 0.9,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Simulates the BLIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            v (np.ndarray): initial voltages of the nodes
            num_iterations (int): amount of iterations that need to be simulated
            dt (float): time step.
            kmin (float): minimal gain of the latch
            kmax (float): maximum gain of the latch
            C (float): capacitor parameter.
            G (float): resistance parameter.
            file (pathlib.Path, None, Optional): absolute path to which data will be logged. If 'None',
                                                 nothing is logged.

        Returns:
            (sample, energy) tuple[np.ndarray, float]: optimal sample and energy.
        """
        N = model.num_variables
        tend = dt * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations)

        # Transform the model to one with no h and mean variance of J
        model.normalize()
        new_model = model.transform_to_no_h()
        J = triu_to_symm(new_model.J)
        model.reconstruct()

        # Add the bias node
        v = np.block([v, 1.0])

        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)
        v += 0.01 * (np.random.random((N + 1,)) - 0.5)

        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        def dvdt(t, vt):
            # Make sure the bias node is 1
            vt[-1] = 1.0

            V_mat = np.array([vt] * vt.shape[0])
            dv = -1 / C * np.sum(J * (V_mat - V_mat.T), axis=0)
            cond1 = (dv > 0) & (vt > 0)
            cond2 = (dv < 0) & (vt < 0)
            dv *= np.where(cond1 | cond2, 1 - vt**2, 1)

            # Make sure the bias node does not change
            dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=np.sign(v),
                model=model,
                num_iterations=num_iterations,
                C=C,
                time_step=dt,
                random_flip=random_flip,
                seed=seed,
                temperature=Temp,
                cooling_rate=r_T,
                stop_criterion=stop_criterion,
            )

            i = 0
            previous_voltages = np.copy(v)
            max_change = np.inf

            while i < (num_iterations) and max_change > stop_criterion:
                tk = t_eval[i]

                # Runge Kutta steps
                k1 = dt * dvdt(tk, previous_voltages)
                k2 = dt * dvdt(tk + 2 / 3 * dt, previous_voltages + 2 / 3 * k1)

                new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2)

                # Do random flipping annealing wise
                if random_flip:
                    rand = np.random.random()
                    if rand < np.exp(-1 / Temp):
                        flip = np.random.choice(N)
                        new_voltages[flip] = -new_voltages[flip]
                Temp *= r_T

                max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                    previous_voltages, ord=np.inf
                )

                # Log everything
                sample = np.sign(new_voltages[:N])
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
