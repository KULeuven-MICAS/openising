import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class DSASolver(SolverBase):
    """ Ising solver based on a discriminatory variation of the simulated annealing algorithm. """

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp: float,
        cooling_rate: float,
        seed: int|None = None,
        file: pathlib.Path|None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Perform optimization using a variation of the simulated annealing algorithm.
        This variation works with cycles where each node must be considered for a flip before
        a node is considered for a flip again. Temperature is lowered between cycles.

        Arguments:
            model: An instance of the IsingModel to be optimized. This defines the energy function.
            initial_state: A 1D numpy array (of 1 and -1's) representing the starting state of the system.
            num_iterations: Number of iterations (steps) for the simulated annealing process.
            initial_temp: Initial temperature for the annealing schedule.
            cooling_rate: Multiplicative factor applied to the temperature after each iteration (0 < cooling_rate < 1).
            seed: (Optional) Seed for the random number generator to ensure reproducibility.
            file: (Optional) Path to an HDF5 file for logging the optimization process. If `None`,
                  no logging is performed.

        Returns:
            A tuple containing:
                - The optimized state as a 1D numpy array.
                - The final energy of the system.
        """
        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() *1000)
        random.seed(seed)

        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,                        # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "change_state": np.bool_,                    # Scalar boolean
            "cycle_started": np.bool_                    # Scalar boolean
        }
        metadata = {
            "solver": "discrimatory_simulated_annealing",
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            "seed": seed
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            logger.write_metadata(**metadata)

            # Setup initial state and energy
            T = initial_temp
            state = initial_state
            energy = model.evaluate(state)

            for _ in range(num_iterations):

                cycle_started = True

                # Iterate over all of the nodes in random order
                nodes = list(range(model.num_variables))
                random.shuffle(nodes)
                for node in nodes:

                    # Obtain new state by flipping that node
                    state[node] = -state[node]

                    # Evaluate the new energy
                    energy_new = model.evaluate(state)

                    # Determine whether to accept the new state
                    delta = energy_new - energy
                    change_state = (delta < 0 or random.random() < np.exp(-delta/T))

                    # Log current iteration data
                    logger.log(energy=energy_new, state=state, change_state=change_state, cycle_started=cycle_started)
                    cycle_started = False

                    # Update the state and energy if the new state is accepted
                    if change_state:
                        energy = energy_new
                    else:
                        state[node] = -state[node]  # Revert the flip if the new state is rejected

                # Decrease the temperature
                T = cooling_rate*T

            # Log the final result
            logger.write_metadata(solution_state=state, solution_energy=energy)

        return state, energy
