import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.clock import clock


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
        clock_freq:float = 1e6,
        clock_op:int = 1000,
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
            clock_freq (float): frequency of the clock cycle
            clock_op (int): amount of operations that can be performed per clock cycle.

        Returns:
            A tuple containing:
                - The optimized state as a 1D numpy array.
                - The final energy of the system.
        """
        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() *1000)
        random.seed(seed)

        clocker = clock(clock_freq, clock_op)
        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,                        # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "change_state": np.bool_,                    # Scalar boolean
            "cycle_started": np.bool_,                    # Scalar boolean
            "time_clock": float
        }
        metadata = {
            "solver": "discrimatory_simulated_annealing",
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            "seed": seed,
            "clock_freq" : clock_freq,
            "clock_op" : clock_op
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            logger.write_metadata(**metadata)

            # Setup initial state and energy
            T = initial_temp
            state = initial_state
            energy = model.evaluate(state)
            operations = 0
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
                    operations += 2*model.num_variables**2
                    # Determine whether to accept the new state
                    delta = energy_new - energy
                    operations += 1
                    change_state = (delta < 0 or random.random() < np.exp(-delta/T))
                    operations += 5
                    # Log current iteration data
                    time = clocker.perform_operations(operations)
                    operations = 0
                    logger.log(energy=energy_new, state=state, change_state=change_state, cycle_started=cycle_started, time_clock=time)
                    cycle_started = False

                    # Update the state and energy if the new state is accepted
                    if change_state:
                        energy = energy_new
                    else:
                        state[node] = -state[node]  # Revert the flip if the new state is rejected

                # Decrease the temperature
                T = cooling_rate*T
                operations += 1

            # Log the final result
            total_time = clocker.get_time()
            logger.write_metadata(solution_state=state, solution_energy=energy, total_time=total_time)

        return state, energy
