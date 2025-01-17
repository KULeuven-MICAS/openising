# import gurobipy as gp
from gurobi_optimods.qubo import solve_qubo
# from gurobipy import GRB
import pathlib
import numpy as np

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger

class Gurobi(SolverBase):
    def convert(self, model:IsingModel) -> np.ndarray:
        """Converts the Ising model to a Gurobi instance.

        Args:
            model (IsingModel): the model that needs to be converted.
        """
        Q, c = model.to_qubo()
        # N = model.num_variables

        # m = gp.Model('ising')
        # x = m.addMVar(shape=N, vtype=GRB.BINARY, name="x")
        # m.setObjective(x.T @ Q @ x + c, GRB.MINIMIZE)
        return Q, c


    def solve(self,
              model: IsingModel,
              file: pathlib.Path|None = None) -> tuple[np.ndarray, float]:
        """Solves the Ising model using Gurobi.

        Args:
            model (IsingModel): the model that needs to be solved
            file (pathlib.Path | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        Q, c = self.convert(model)
        # if file is not None:
        #     Gur_model.Params.LogFile = file

        # Gur_model.optimize()
        result = solve_qubo(Q)
        self.convert_logger(file, result, c)
        return result.solution, result.objective_value + c

    def convert_logger(self, file:pathlib.Path, result, c) -> None:
        """Converts the Gurobi logfile to a HDF5 one.

        Args:
            file (pathlib.Path): path to the logfile
            result: the result of the Gurobi optimization
            c: constant value of the transformation of Ising to QUBO form.
        """
        with HDF5Logger(file) as logger:
            logger.write_metadata(solution_state=result.solution, solution_energy=result.objective_value + c)
