from abc import ABC, abstractmethod
import itertools
import pathlib
import csv

from ising.model import BinaryQuadraticModel
from ising.typing import Bias
from ising.utils.convert import SampleLike

class SolverLogger:

    def __init__(self, file: pathlib.Path|None, bqm: BinaryQuadraticModel):
        self.file = file
        self.variables = list(bqm.linear.keys())
        self.csv_writer = None

    def __enter__(self):
        if self.file:
            self.file = pathlib.Path(self.file).open(mode='w', newline='')
            self.csv_writer = csv.writer(self.file)
            self.csv_writer.writerow(itertools.chain(["time","energy"], self.variables))
        return self

    def write(self, time: float, energy: Bias, sample: SampleLike):
        if self.csv_writer:
            self.csv_writer.writerow(itertools.chain([time, energy], [sample[v] for v in self.variables]))

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.file:
            self.file.close()


class Solver(ABC):
    """
    Abstract Base Class for Ising solvers.
    """
    @abstractmethod
    def solve(self, bqm: BinaryQuadraticModel):
        pass

    def open_log(file: pathlib.Path|None, bqm: BinaryQuadraticModel):
        return SolverLogger(file, bqm)
