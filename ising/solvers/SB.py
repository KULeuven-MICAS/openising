import numpy as np
import pathlib
from abc import abstractmethod

from ising.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger


class SB(SolverBase):
    """Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    This class inherits from the abstract Solver base class.
    """

    def update_x(self, y, dt, a0, node):
        return a0 * y[node] * dt

    def update_rule(self, x, y, node):
        x[node] = np.sign(x[node])
        y[node] = 0.0
        return x, y

    @abstractmethod
    def solve(self, model: IsingModel):
        pass


class ballisticSB(SB):
    def solve(
        self,
        model: IsingModel,
        x: np.ndarray,
        y: np.ndarray,
        num_iterations: int,
        at: callable,
        c0: float,
        dt: float,
        a0: float=1.,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """Performs the ballistic Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation introduces perfectly inelastic walls at |x_i| = 1
        to reduce analog errors.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            x (np.ndarray): the initial position of the nonlinear oscillators.
            y (np.ndarray): the initial momenta of the nonlinear oscillators.
            num_iterations (int): amount of iterations that needs to be performed.
            a0 (float, Optional): hyperparameter. Defaults to 1.
            at (callable): changing hyperparameter that induces the bifurcation.
            c0 (float): hyperparameter.
            dt (float): time step.
            file (pathlib.Path, None, Optional): full path to which data will be logged. If 'None',
                                                 no logging is performed

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        tk = 0.0
        N = model.num_variables

        schema = {
            "time": float,
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.int32, (N,)),
            "momenta": (np.int32, (N,)),
            "at": np.float32,
        }

        metadata = {
            "solver": "ballistic_simulated_bifurcation",
            "time_step": dt,
            "a0": a0,
            "c0": c0,
            "num_iterations": num_iterations,
        }
        with HDF5Logger(file, schema) as log:
            log.write_metadata(**metadata)
            for i in range(num_iterations):
                atk = at(tk)
                for j in range(N):
                    y[j] += (-(a0 - atk) * x[j] + c0 * np.dot(model.J[:, j], x) + c0 * model.h[j]) * dt
                    x[j] += self.update_x(y, dt, a0, j)
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=at(tk))
                tk += dt
            log.write_metadata(solution_state=sample, solution_energy=energy)
        return sample, energy


class discreteSB(SB):
    def solve(
        self,
        model: IsingModel,
        x: np.ndarray,
        y: np.ndarray,
        num_iterations: int,
        at: callable,
        c0: float,
        dt: float,
        a0: float=1.,
        file: pathlib.Path|None=None,
    )-> tuple[np.ndarray, float]:
        """Performs the discrete Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation discretizes the positions x_i at all times to reduce analog errors.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            x (np.ndarray): the initial position of the nonlinear oscillators.
            y (np.ndarray): the initial momenta of the nonlinear oscillators.
            num_iterations (int): amount of iterations that needs to be performed.
            a0 (float, Optional): hyperparameter. Defaults to 1.
            at (callable): changing hyperparameter that induces the bifurcation.
            c0 (float): hyperparameter.
            dt (float): time step.
            file (pathlib.Path, None, Optional): full path to which data will be logged. If 'None',
                                                 no logging is performed

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        N = model.num_variables
        tk = 0.0

        schema = {
            "time": float,
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.int32, (N,)),
            "momenta": (np.int32, (N,)),
            "at": np.float32,
        }

        metadata = {
            "solver": "discrete_simulated_bifurcation",
            "time_step": dt,
            "a0": a0,
            "c0": c0,
            "num_iterations": num_iterations,
        }

        with HDF5Logger(file, schema) as log:
            log.write_metadata(metadata)
            for i in range(num_iterations):
                atk = at(tk)
                for j in range(N):
                    y[j] += (-(a0 - atk) * x[j] + c0 * np.inner(model.J[:, j], np.sign(x)) + c0 * model.h[j]) * dt
                    x[j] += self.update_x(y, dt, a0, j)
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk)
                tk += dt
            log.write_metadata(solution_state=sample, solution_energy=energy)
        return sample, energy
