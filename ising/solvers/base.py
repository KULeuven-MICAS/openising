from abc import ABC, abstractmethod
import numpy as np

from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


class SolverBase(ABC):
    """Abstract Base Class for Ising solvers."""

    def __init__(self):
        self.name= ""


    @abstractmethod
    def solve(self, model: IsingModel):
        pass

    def log_metadata(
        self,
        logger: HDF5Logger,
        initial_state: np.ndarray,
        model: IsingModel,
        num_iterations: int,
        **kwargs,
    ):
        """Log metadata to the HDF5 file. This metadata is mostly needed for postprocessing and
            thus common to most solvers.

        Args:
            logger (HDF5Logger): the logger object to use for logging metadata
            initial_state (np.ndarray): the initial state of the system
            model (IsingModel): the model to be optimized
            num_iterations (int): the number of iterations to run the solver for
        """
        metadata = {
            "solver": self.name,
            "num_variables": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
        logger.write_metadata(**metadata)
