import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class InSituSASolver(SolverBase):
    """Ising solver based on the classical simulated annealing algorithm."""

    def __init__(self):
        self.name = "InSituSA"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp: float,
        cooling_rate: float,
        nb_flips: float,
        seed: int | None = None,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Perform optimization using the classical simulated annealing algorithm.

        @param model: An instance of the IsingModel to be optimized. This defines the energy function.
        @param initial_state: A 1D numpy array (of 1 and -1's) representing the starting state of the system.
        @param num_iterations: Number of iterations (steps) for the simulated annealing process.
        @param initial_temp: Initial temperature for the annealing schedule.
        @param cooling_rate: Multiplicative factor applied to the temperature after each iteration.
        @param seed: Seed for the random number generator to ensure reproducibility.
        @param nb_flips: percentage of total nodes that are flipped each iteration.
        @param file: Path to an HDF5 file for logging the optimization process. If `None`, no logging is performed.

        @return state: The final state of the system.
        @return energy: optimal energy the solver reaches.
        """

        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        coupling = np.diag(model.h) + model.J + model.J.T
        nb_flips = int(nb_flips * model.num_variables)

        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,  # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            if logger.filename is not None:
                logger.write_metadata(
                    initial_state=initial_state,
                    model_name=self.name,
                    problem_size=model.num_variables,
                    num_iterations=num_iterations,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate,
                    nb_flips=nb_flips,
                    seed=seed,
                )

            start_time = time.time()

            # Setup initial state and energy
            Temp = initial_temp
            state = np.sign(initial_state, dtype=np.float32)

            for _ in range(num_iterations):
                # Select a random node to flip
                sigma_f = np.where(
                    np.isin(
                        np.arange(model.num_variables),
                        np.random.choice(model.num_variables, size=(nb_flips,), replace=False),
                    ),
                    np.float32(1),
                    np.float32(0))
                # Obtain new state by flipping that node
                sigma_new = state * (1 - 2 * sigma_f)
                sigma_c = sigma_new * sigma_f
                sigma_r = sigma_new * (1 - sigma_f)

                # Evaluate the new energy
                f_T = 1 / (-0.006 * Temp + 5) - 0.2
                delta = sigma_r.T @ coupling @ sigma_c * f_T

                # Determine whether to accept the new state (Metropolis)
                change_state = delta < 0 or random.random() >= delta
                state = sigma_new if change_state else state  # accept correct state for flip

                # Log current iteration data
                if logger.filename is not None:
                    logger.log(energy=model.evaluate(state), state=state)
                # Decrease the temperature
                Temp *= cooling_rate

            end_time = time.time()
            nb_operations = num_iterations * (nb_flips + 6 * model.num_variables + 3 * nb_flips**2 + 1)
            if logger.filename is not None:
                logger.write_metadata(
                    solution_state=state, solution_energy=model.evaluate(state), total_operations=nb_operations
                )
            else:
                energy = model.evaluate(state)

        return state, energy, end_time - start_time, nb_operations
