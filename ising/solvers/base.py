from abc import ABC, abstractmethod

from ising.model.ising import IsingModel

class SolverBase(ABC):
    """ Abstract Base Class for Ising solvers. """
    @abstractmethod
    def solve(self, model: IsingModel):
        pass
