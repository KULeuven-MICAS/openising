import random
import pathlib
import time
import numpy as np

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class CIMSolver(SolverBase):
    """Ising solver based on the classical simulated annealing algorithm."""

    def __init__(self):
        self.name = "CIM"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtCIM: float,
        zeta: float = 1.0,
        seed: int | None = None,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Perform optimization using the classical simulated annealing algorithm.

        @param model: An instance of the IsingModel to be optimized. This defines the energy function.
        @param initial_state: A 1D numpy array (of 1 and -1's) representing the starting state of the system.
        @param num_iterations: Number of iterations (steps) for the simulated annealing process.
        @param zeta: Displacement parameter.
        @param seed: Seed for the random number generator to ensure reproducibility.
        @param file: Path to an HDF5 file for logging the optimization process. If `None`, no logging is performed.

        @return state: the optimized state as a 1D numpy array.
        @return energy: the final energy of the system.
        """

        # seed the random number generator. Use a timestamp-based seed if non is provided.
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        if np.linalg.norm(model.h) >= 1e-6:
            use_bias = True
            new_model = model.transform_to_no_h()
        else:
            use_bias = False
            new_model = model
        coupling = (new_model.J + new_model.J.T).astype(np.float32)
        zeta = np.float32(zeta)
        dtCIM = np.float32(dtCIM)

        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,  # Scalar float
            "state": (np.int8, (model.num_variables,)),  # Vector of int8 (to hold -1 and 1)
            "x": (np.float32, (model.num_variables,)),  # Vector of float32 (to hold pulses)
        }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            logger.write_metadata(
                initial_state=initial_state,
                model_name=self.name,
                problem_size=model.num_variables,
                num_iterations=num_iterations,
                zeta=zeta,
                dtCIM=dtCIM,
                seed=seed,
            )

            # Setup initial state and energy
            state = initial_state.astype(np.float32)
            energy = model.evaluate(state)
            x = np.float32(0.1) * state if not use_bias else np.block([np.float32(0.1) * state, np.float32(1)])
            if logger.filename is not None:
                logger.log(energy=energy, state=state, x=x)

            for _ in range(num_iterations):
                x += dtCIM * (
                    self.pump_loss_law(_, num_iterations) * x
                    + zeta * coupling @ x
                    + np.random.normal(size=x.shape).astype(np.float32)
                )
                x[-1] = np.float32(1) if use_bias else x[-1]

                # Account for saturation
                x = np.where(np.abs(x) > 1, np.sign(x), x)

                if logger.filename is not None:
                    # Log the current state
                    energy = model.evaluate(np.sign(x[: model.num_variables]))
                    logger.log(energy=energy, state=np.sign(x[: model.num_variables]), x=x)

            logger.write_metadata(solution_state=state, solution_energy=energy)

        return state, energy

    def pump_loss_law(self, it: int, num_iterations: int):
        return np.float32(16 / (1 + np.exp(np.log(1 / 961) / num_iterations * it + np.log(31))) - 15.5)
