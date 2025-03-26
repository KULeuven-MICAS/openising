import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.clock import clock


class SASolver(SolverBase):
    """Ising solver based on the classical simulated annealing algorithm."""

    def __init__(self):
        self.name = "SA"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp: float,
        cooling_rate: float,
        seed: int | None = None,
        file: pathlib.Path | None = None,
        clock_freq: float = 1e6,
        clock_op: int = 1000,
    ) -> tuple[np.ndarray, float]:
        """
        Perform optimization using the classical simulated annealing algorithm.

        Arguments:
            model: An instance of the IsingModel to be optimized. This defines the energy function.
            initial_state: A 1D numpy array (of 1 and -1's) representing the starting state of the system.
            num_iterations: Number of iterations (steps) for the simulated annealing process.
            initial_temp: Initial temperature for the annealing schedule.
            cooling_rate: Multiplicative factor applied to the temperature after each iteration (0 < cooling_rate < 1).
            seed: (Optional) Seed for the random number generator to ensure reproducibility.
            file: (Optional) Path to an HDF5 file for logging the optimization process. If `None`,
                  no logging is performed.
            clock_freq (float): frequency of the clock cycle
            clock_op (int): amount of operations that can be performed per clock cycle.

        Returns:
            A tuple containing:
                - The optimized state as a 1D numpy array.
                - The final energy of the system.
        """

        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)
        clocker = clock(frequency=clock_freq, nb_operations=clock_op)

        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,  # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "change_state": np.bool_,  # Scalar boolean
            "time_clock": float,
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            self.log_metadata(
                logger=logger,
                initial_state=initial_state,
                model=model,
                num_iterations=num_iterations,
                initial_temp=initial_temp,
                cooling_rate=cooling_rate,
                seed=seed,
            )

            # Setup initial state and energy
            T = initial_temp
            state = np.sign(initial_state)
            energy = model.evaluate(state)
            clocker.add_operations(2 * model.num_variables**2)
            clocker.perform_operations()
            for _ in range(num_iterations):
                # Select a random node to flip
                node = random.randrange(0, model.num_variables)

                # Obtain new state by flipping that node
                state[node] = -state[node]

                # Evaluate the new energy
                energy_new = model.evaluate(state)
                clocker.add_cycles(1 + np.log2(model.num_variables))

                delta = energy_new - energy
                clocker.add_operations(1)

                # Determine whether to accept the new state (Metropolis)
                change_state = delta < 0 or random.random() < np.exp(-delta / T)
                clocker.add_operations(5)

                # Log current iteration data
                current_time = clocker.perform_operations()
                logger.log(energy=energy_new, state=state, change_state=change_state, time_clock=current_time)

                # Update the state and energy if the new state is accepted
                if change_state:
                    energy = energy_new
                else:
                    state[node] = -state[node]  # Revert the flip if the new state is rejected

                # Decrease the temperature
                T = cooling_rate * T
                clocker.add_operations(1)
                clocker.perform_operations()

            # Log the final result
            current_time = clocker.perform_operations()
            # logger.log(energy=energy_new, state=state, change_state=change_state, time_clock=current_time)
            final_time = clocker.get_time()
            nb_operations = (
                num_iterations * (3 * model.num_variables**2 + 2 * model.num_variables + 8)
                + 3 * model.num_variables**2
                + 2 * model.num_variables
            )
            logger.write_metadata(
                solution_state=state, solution_energy=energy, total_time=final_time, total_operations=nb_operations
            )

        return state, energy
