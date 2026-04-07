import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


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
        cooling_rate_SA: float,
        seed: int | None = None,
        stop_criterion:bool = False,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Perform optimization using the classical simulated annealing algorithm.

        @param model: An instance of the IsingModel to be optimized. This defines the energy function.
        @param initial_state: A 1D numpy array representing the starting state of the system.
        @param num_iterations: Number of iterations for the simulated annealing process.
        @param initial_temp: Initial temperature for the annealing schedule.
        @param cooling_rate: Multiplicative factor applied to the temperature after each iteration.
        @param seed: Seed for the random number generator to ensure reproducibility.
        @param file: Path to an HDF5 file for logging the optimization process. If `None`, no logging is performed.

        @return state: The optimized state as a 1D numpy array.
        @return energy: The final energy of the system.
        """

        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        if not stop_criterion:
            self.zero_en_length = num_iterations

        # Set up schema and metadata for logging
        schema = {
            "time": np.float32,  # Scalar float
            "energy": np.float32,  # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "change_state": np.bool_,  # Scalar boolean
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            if logger.filename is not None:
                self.log_metadata(
                    logger=logger,
                    initial_state=initial_state,
                    model=model,
                    num_iterations=num_iterations,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate_SA,
                    seed=seed,
                )
            k = 0
            current_length = 0
            start_time = time.time()

            # Setup initial state and energy
            T = initial_temp
            state = np.sign(initial_state).astype(np.float32)
            energy = model.evaluate(state)

            if logger.filename is not None:
                logger.log(time=0.0, energy=energy, state=state, change_state=False)
            while k < num_iterations and current_length < self.zero_en_length:
                # Select a random node to flip
                node = random.randrange(0, model.num_variables)  # 1

                # Obtain new state by flipping that node
                state[node] = -state[node]  # 1

                # Evaluate the new energy
                energy_new = model.evaluate(state)  # 2N**2+4*N

                delta = energy_new - energy  # 1

                # Determine whether to accept the new state (Metropolis)
                change_state = delta < 0 or random.random() < np.exp(-delta / T)  # 5

                # Update the state and energy if the new state is accepted
                if change_state:  # 2
                    energy = energy_new
                    current_length = 0
                else:
                    state[node] = -state[node]  # Revert the flip if the new state is rejected
                    current_length += 1 if stop_criterion else 0
                # Decrease the temperature
                T = cooling_rate_SA * T  # 1
                k += 1
                # Log current iteration data
                if logger.filename is not None:
                    time_elapsed = time.time() - start_time
                    logger.log(time=time_elapsed, energy=energy_new, state=state, change_state=change_state)

            nb_operations = (
                num_iterations * (2 * model.num_variables**2 + 4 * model.num_variables + 8)
                + 2 * model.num_variables**2
                + 4 * model.num_variables
            )
            if logger.filename is not None:
                logger.write_metadata(
                    total_time=time_elapsed,
                    solution_state=state,
                    solution_energy=energy,
                    total_operations=nb_operations,
                )
            else:
                time_elapsed = time.time() - start_time

        return state, energy, time_elapsed, nb_operations, k
