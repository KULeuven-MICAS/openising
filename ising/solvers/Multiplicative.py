import numpy as np
import pathlib
import time

from ising.flow import LOGGER, TOP
from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

    def Ka(self, time:float, end_time:float)->float:
        """Returns the coupling annealing term.

        Args:
            time (float): the time.
            end_time (float): the end time.
        Returns:
            Ka (float): the coupling annealing term.
        """
        return 1-np.exp(-time/end_time)


    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        seed: int = 0,
        initial_temp_cont: float = 1.0,
        end_temp_cont: float = 0.05,
        stop_criterion: float = 1e-8,
        coupling_annealing: bool = False,
        file: pathlib.Path | None = None,
    ) -> tuple[float, np.ndarray]:
        """Solves the given problem using a multiplicative coupling scheme.

        Args:
            model (IsingModel): the model to solve.
            initial_state (np.ndarray): the initial spins of the nodes.
            dtMult (float): time step.
            num_iterations (int): the number of iterations.
            seed (int, optional): the seed for random number generation. Defaults to 0.
            initial_temp_cont (float, optional): the initial temperature for the additive voltage noise.
                                                 Defaults to 1.0.
            end_temp_cont (float, optional): the final temperature for the additive voltage noise. Defaults to 0.05.
            stop_criterion (float, optional): the stopping criterion to stop the solver when the voltages don't change
                                              too much anymore. Defaults to 1e-8.
            coupling_annealing (bool, optional): whether to anneal the coupling matrix. Defaults to False.
            file (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: the best energy and the best sample.
        """
        # Set up the time evaluations
        tend = dtMult * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            zero_h = False
        else:
            new_model = model
            zero_h = True
        J = triu_to_symm(new_model.J)
        lam, _ = np.linalg.eig(J + np.eye(J.shape[0]))
        LOGGER.debug("Maximal eigenvalue of J + I: ", lam[0])

        # make sure the correct random seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the bias node and add noise to the initial voltages
        N = model.num_variables
        initial_state = np.loadtxt(TOP / "ising/flow/000.txt")[:N]
        if not zero_h:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state

        # Schema for logging
        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        # Define the system equations
        def dvdt(t: float, vt: np.ndarray, coupling: np.ndarray):
            """Differential equations for the multiplicative BRIM model.

            Args:
                t (float): time
                vt (np.ndarray): current voltages
                coupling (np.ndarray): coupling matrix J

            Returns:
                dv (np.ndarray): the change of the voltages
            """

            # set bias node to 1.
            if not zero_h:
                vt[-1] = 1.0

            # ZIV diode
            # z = (vt + 1)*(vt-1)*vt

            # Compute the voltage change dv
            dv = np.dot(coupling, vt) #- z

            # Ensure the voltages stay in the range [-1, 1]
            cond1 = (dv > 0) & (vt > 1)
            cond2 = (dv < 0) & (vt < -1)
            dv *= np.where(cond1 | cond2, 0.0, 1.)

            # Ensure the bias node does not change
            if not zero_h:
                dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=np.sign(v[:-1]),
                model=model,
                num_iterations=num_iterations,
                time_step=dtMult,
                temperature=initial_temp_cont,
                coupling_annealing=coupling_annealing
            )

            # Set up the simulation
            i = 0
            max_change = np.inf
            Temp = initial_temp_cont if initial_temp_cont < 1.0 else 0.5
            cooling_rate = (
                (end_temp_cont / initial_temp_cont) ** (1 / (num_iterations - 1)) if initial_temp_cont != 0.0 else 1.0
            )
            previous_voltages = np.copy(v)

            while i < num_iterations and max_change > stop_criterion:
                tk = t_eval[i]

                if coupling_annealing:
                    Ka = self.Ka(tk, tend)
                else:
                    Ka = 1.0

                # Runge Kutta steps, k1 is the derivative at time step t, k2 is the derivative at time step t+2/3*dt
                k1 = dtMult * dvdt(tk, previous_voltages, Ka*J)
                k2 = dtMult * dvdt(tk + 2 / 3 * dtMult, previous_voltages + 2 / 3 * k1, Ka*J)

                # Add noise and update the voltages
                if Temp != 0.0:
                    noise = Temp * (np.random.normal(scale=1 / 1.96, size=previous_voltages.shape))
                    cond1 = (previous_voltages >= 1) & (noise > 0)
                    cond2 = (previous_voltages <= -1) & (noise < 0)
                    noise *= np.where(cond1 | cond2, 0.0, 1.0)
                else:
                    noise = np.zeros_like(previous_voltages)
                new_voltages = previous_voltages + 1.0 / 4.0 * (k1 + 3.0 * k2) + noise
                Temp *= cooling_rate

                # Log everything
                sample = np.sign(new_voltages[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

                # Update the criterion changes
                if i > 0:
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

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy
