from abc import ABC, abstractmethod
from collections.abc import Iterable
import pathlib
import csv
import itertools

from ising.model.ising import IsingModel

class SolverLogger:

    def __init__(self, file: pathlib.Path|None, model: IsingModel, *extra_fields):
        self.file = file
        self.csv_writer = None
        self.header = itertools.chain(["time", "energy"], range(model.num_variables), extra_fields)

    def __enter__(self):
        if self.file:
            self.file = pathlib.Path(self.file).open(mode='w', newline='')
            self.csv_writer = csv.writer(self.file)
            self.csv_writer.writerow(self.header)
        return self

    def write(self, time, energy, sample, *extra_fields):
        if self.csv_writer:
            fields = itertools.chain(*(iter(arg) if isinstance(arg, Iterable) else (arg,) for arg in extra_fields))
            self.csv_writer.writerow(itertools.chain([time, energy], sample, fields))

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.file:
            self.file.close()


class Solver(ABC):
    """
    Abstract Base Class for Ising solvers.
    """
    @abstractmethod
    def solve(self, model: IsingModel):
        pass

    def open_log(self, file: pathlib.Path|None, model: IsingModel, *extra_fields):
        return SolverLogger(file, model, *extra_fields)

    def change_node(self, node:int):
        self.sigma[node] = -self.sigma[node]
