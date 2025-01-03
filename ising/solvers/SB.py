import numpy as np
import pathlib
from abc import abstractmethod

from ising.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.clock import clock


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
        clock_freq:float=1e6,
        clock_op:int=1000
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
            clock_freq (float): frequency of the clock cycle
            clock_op (int): amount of operations that can be performed per clock cycle.

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        tk = 0.0
        N = model.num_variables
        J = triu_to_symm(model.J)
        clocker = clock(clock_freq, clock_op)

        schema = {
            "time": float,
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.int32, (N,)),
            "momenta": (np.int32, (N,)),
            "at": np.float32,
            "time_clock": float
        }

        metadata = {
            "solver": "ballistic_simulated_bifurcation",
            "time_step": dt,
            "a0": a0,
            "c0": c0,
            "num_iterations": num_iterations,
            "clock_freq": clock_freq,
            "clock_op": clock_op
        }

        with HDF5Logger(file, schema) as log:
            log.write_metadata(**metadata)
            for _ in range(num_iterations):
                atk = at(tk)
                operations = 1
                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0*model.h) * dt

                operations += 2*N**2 + 5*N
                clocker.perform_operations(operations)

                x += a0*y*dt
                operations = 2*N + 1
                clocker.perform_operations(operations)

                operations = 0
                for j in range(N):
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                        operations += 2

                time = clocker.perform_operations(operations+1)
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk, time_clock=time)
                tk += dt
            total_time = clocker.get_time()
            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=total_time)
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
        clock_freq:float=1e6,
        clock_op:int=1000
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
            clock_freq (float): frequency of the clock cycle
            clock_op (int): amount of operations that can be performed per clock cycle.

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        N = model.num_variables
        tk = 0.0
        J = triu_to_symm(model.J)
        clocker = clock(clock_freq, clock_op)

        schema = {
            "time": float,
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.int32, (N,)),
            "momenta": (np.int32, (N,)),
            "at": np.float32,
            "time_clock": float
        }

        metadata = {
            "solver": "discrete_simulated_bifurcation",
            "time_step": dt,
            "a0": a0,
            "c0": c0,
            "num_iterations": num_iterations,
            "clock_freq": clock_freq,
            "clock_op": clock_op
        }

        with HDF5Logger(file, schema) as log:
            log.write_metadata(**metadata)
            for _ in range(num_iterations):
                atk = at(tk)
                operations = 1
                y += (-(a0 - atk) * x + c0 * np.matmul(J, np.sign(x)) + c0*model.h) * dt
                operations += 2*N**2 + 5*N
                clocker.perform_operations(operations)
                x += a0*y*dt
                operations = 2*N
                clocker.perform_operations(operations)
                operations = 1
                for j in range(N):
                #     y[j] += (-(a0 - atk) * x[j] + c0 * np.inner(model.J[:, j], np.sign(x)) + c0 * model.h[j]) * dt
                #     x[j] += self.update_x(y, dt, a0, j)
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                        operations += 2
                sample = np.sign(x)
                energy = model.evaluate(sample)
                time = clocker.perform_operations(operations)
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk, time_clock=time)
                tk += dt

            total_time = clocker.get_time()
            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=total_time)
        return sample, energy
