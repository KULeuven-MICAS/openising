import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class InSituSASolver(SolverBase):
    """Ising solver based on the classical simulated annealing algorithm. This implementation is based on the paper of\
          [Qian et al.](https://ieeexplore.ieee.org/document/11133307)"""

    def __init__(self):
        super().__init__()
        self.name = "InSituSA"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp_inSituSA: float,
        cooling_rate_inSituSA: float,
        nb_flips: int = -1,
        seed: int | None = None,
        stop_criterion: bool = False,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float, float, int, int]:
        """
        Perform optimization using the classical simulated annealing algorithm.

        @type model: IsingModel
        @param model: An instance of the IsingModel to be optimized. This defines the energy function.
        @type initial_state: np.ndarray
        @param initial_state: the starting state of the system.
        @type num_iterations: int
        @param num_iterations: Number of iterations for the simulated annealing process.
        @type initial_temp_inSituSA: float
        @param initial_temp_inSituSA: Initial temperature for the annealing schedule.
        @type cooling_rate_inSituSA: float
        @param cooling_rate_inSituSA: Multiplicative factor applied to the temperature after each iteration.
        @type nb_flips: int, optional
        @param nb_flips: total amount of nodes that are flipped each iteration. If -1, the default value of 2 is used.
        @type seed: int, None, optional
        @param seed: Seed for the random number generator to ensure reproducibility.
        @type stop_criterion: bool, optional
        @param stop_criterion: Whether to stop the solver when the energy converges. Defaults to False.
        @type file: pathlib.Path, None, optional
        @param file: Path to an HDF5 file for logging the optimization process. If `None`, no logging is performed.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: optimal state, optimal energy, simulation time, amount of operations, performed iterations.
        """
        if nb_flips == -1:
            nb_flips = 2

        if not stop_criterion:
            self.zero_en_length = num_iterations

        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        coupling = -(np.diag(model.h) + model.J + model.J.T)

        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,  # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "time": np.float32,
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            if logger.filename is not None:
                self.log_metadata(
                    logger=logger,
                    initial_state=initial_state,
                    model=model,
                    num_iterations=num_iterations,
                    initial_temp=initial_temp_inSituSA,
                    cooling_rate=cooling_rate_inSituSA,
                    nb_flips=nb_flips,
                    seed=seed,
                )
            k = 0
            current_length = 0
            start_time = time.time()

            # Setup initial state and energy
            Temp = initial_temp_inSituSA
            state = np.sign(initial_state, dtype=np.float32)
            energy = model.evaluate(state)
            if logger.filename is not None:
                logger.log(time=0.0, energy=energy, state=state)
            while k < num_iterations and current_length < self.zero_en_length:
                # Select a random node to flip
                sigma_f = np.where(
                    np.isin(
                        np.arange(model.num_variables),
                        np.random.choice(model.num_variables, size=(nb_flips,), replace=False),
                    ),
                    np.float32(1),
                    np.float32(0),
                )
                # Obtain new state by flipping that node
                sigma_new = state * (1 - 2 * sigma_f)
                sigma_c = sigma_new * sigma_f
                sigma_r = sigma_new * (1 - sigma_f)

                # Evaluate the new energy
                f_T = 1 / (-0.006 * Temp + 5) - 0.2
                delta = sigma_r.T @ coupling @ sigma_c * f_T
                rand = np.random.uniform(0, 1)
                # Determine whether to accept the new state (Metropolis)
                if delta <= 0:
                    state = sigma_new
                    current_length = 0
                elif delta >= rand:
                    state = sigma_new
                    current_length = 0
                else:
                    current_length += 1 if stop_criterion else 0
                # state = sigma_new if change_state else state  # accept correct state for flip

                # Decrease the temperature
                Temp *= cooling_rate_inSituSA
                k += 1
                # Log current iteration data
                if logger.filename is not None:
                    elapsed_time = time.time() - start_time
                    logger.log(energy=model.evaluate(state), state=state, time=elapsed_time)

            nb_operations = num_iterations * (
                nb_flips + 6 * model.num_variables + (model.num_variables - nb_flips) * nb_flips + 1
            )
            energy = model.evaluate(state)
            if logger.filename is not None:
                logger.write_metadata(
                    solution_state=state,
                    solution_energy=energy,
                    total_operations=nb_operations,
                    total_time=elapsed_time,
                )
            else:
                elapsed_time = time.time() - start_time

        return state, energy, elapsed_time, nb_operations, k
