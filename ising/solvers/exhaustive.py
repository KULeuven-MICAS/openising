import pathlib
import numpy as np

from ising.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger


class ExhaustiveSolver(SolverBase):
    """ Exhaustive Ising solver. """

    def solve(
        self,
        model: IsingModel,
        file: pathlib.Path|None = None
    ) -> tuple[np.ndarray, float]:
        """
        Iteratively search through the state space for the best solution.

        Arguments:
            model: An instance of the IsingModel to be optimized. This defines the energy function.
            file: (Optional) Path to an HDF5 file for logging the optimization process. If `None`,
                  no logging is performed.

        Returns:
            A tuple containing:
                - The optimized state as a 1D numpy array.
                - The final energy of the system.
        """
        # Set up schema and metadata for logging
        schema = {
            "energy": np.float32,                       # Scalar float
            "state": (np.int8, (model.num_variables,))  # Vector of int8 (to hold -1 and 1)
        }
        metadata = { "solver": "exhaustive" }

        # Initialize logger
        with HDF5Logger(file, schema) as logger:
            logger.write_metadata(**metadata)

            # Initial setup
            state = np.ones(model.num_variables, dtype=np.int8)
            solution_energy = model.evaluate(state)
            solution_state = state
            logger.log(energy=solution_energy, state=solution_state)

            # Iterate over all to be flipped nodes to generate the Gray sequence
            for node in self._gray_code_generator(model.num_variables):

                # Obtain new state by flipping that node
                state[node] = -state[node]

                # Evaluate the new energy
                energy = model.evaluate(state)

                # Update logs
                logger.log(energy=energy, state=state)

                # Update the state and energy if the new state is accepted
                if energy < solution_energy:
                    solution_energy = energy
                    solution_state = state.copy()


            # Log the final result
            logger.write_metadata(solution_state=solution_state, solution_energy=solution_energy)

        return solution_state, solution_energy


    def _gray_code_generator(self, n):
        """ Generates the indices of the bit to flip in the Gray code sequence for a given number of bits `n`. """
        previous = 0

        # Generate the Gray code sequence for n bits
        for i in range(1, 2**n):

            # Compute the next Gray code value
            current = i ^ (i >> 1)

            # The bit that changes is the index of the most significant bit that differs
            # between the previous and current Gray code values
            diff = previous ^ current
            bit_to_flip = diff.bit_length() - 1  # Index of the highest bit that is different

            yield bit_to_flip

            # Update previous for the next iteration
            previous = current
