import numpy as np
import pathlib
from abc import abstractmethod

from ising.stages.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class SB(SolverBase):
    """Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    This class inherits from the abstract Solver base class.
    """

    def __init__(self):
        self.name = "SB"

    def update_x(self, y, dt, a0):
        return a0 * y * dt

    def update_rule(self, x, y, node):
        x[node] = np.sign(x[node])
        y[node] = 0.0


    def at(self, t, a0, dt, num_iterations) -> float:
        return 2*a0 / (dt*num_iterations) * t

    def cast_to_values(self, casted_values, actual_values):
        return np.array([actual_values[np.argmin(np.abs(actual_values - xi))] for xi in casted_values])

    @abstractmethod
    def solve(self, model: IsingModel):
        pass


class ballisticSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"b{self.name}"

    def solve(
        self,
        model:          IsingModel,
        initial_state:  np.ndarray,
        num_iterations: int,
        c0:             float,
        dtSB:           float,
        a0:             float = 1.0,
        file:           pathlib.Path | None = None,
        bit_width_x:      int = 8,
        bit_width_y:      int = 8,
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
            bit_width (int, optional): The bit width for the position and momenta. Defaults to 16.

        Returns:
            sample, energy (tuple[np.ndarray, float]): optimal solution and energy
        """
        N  = model.num_variables

        # Set up the model and initial states with the correct data type
        # x_values        = np.linspace(-1, 1, 2**(bit_width_x)-1)
        # y_values        = np.linspace(-1, 1, 2**(bit_width_y)-1)
        J             = np.array(triu_to_symm(model.J), dtype=np.float32)
        h             = np.array(model.h)
        initial_state = np.array(initial_state)
        x             = np.zeros_like(initial_state, dtype=np.float32)
        y             = np.random.uniform(-0.1, 0.1, (model.num_variables, ))

        schema = {
            "energy"    : float,
            "state"     : (np.int8, (N,)),
            "positions" : (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger         = log,
                initial_state  = np.sign(x),
                model          = model,
                num_iterations = num_iterations,
                time_step      = dtSB,
                a0             = a0,
                c0             = c0,
            )

            sample = np.sign(x)
            energy = model.evaluate(sample)
            tk   = 0.0
            log.log(energy=energy, state=sample, positions=x)
            for _ in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations)

                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0 *  h) * dtSB
                # y = self.cast_to_values(y, y_values)
                x += self.update_x(y, dtSB, a0)
                # x = self.cast_to_values(x, x_values)

                y = np.where(np.abs(x) >= 1, 0, y)
                x = np.where(np.abs(x) >= 1, np.sign(x), x)

                sample = np.sign(x)
                energy = model.evaluate(sample)
                tk    += dtSB
                log.log(energy=energy, state=sample, positions=x)

            nb_operations = num_iterations * (2 * N**2 + 10 * N + 3)
            log.write_metadata(
                solution_state=sample, solution_energy=energy, total_operations=nb_operations
            )
        return sample, energy


class discreteSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"d{self.name}"

    def solve(
        self,
        model: IsingModel,
        initial_state:np.ndarray,
        num_iterations: int,
        c0: float,
        dtSB: float,
        a0: float = 1.0,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
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
        J = triu_to_symm(model.J)

        x = 0.01 * initial_state
        y = np.zeros_like(x)

        schema = {
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=np.sign(x),
                model=model,
                num_iterations=num_iterations,
                time_step=dtSB,
                a0=a0,
                c0=c0,
            )
            sample = np.sign(x)
            energy = model.evaluate(sample)
            log.log(energy=energy, state=sample, positions=x)
            for i in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations)

                y += (-(a0 - atk) * x + c0 * np.matmul(J, np.sign(x)) + c0 * model.h) * dtSB
                x += self.update_x(y, dtSB, a0)

                for j in range(N):
                    if np.abs(x[j]) > 1:
                        self.update_rule(x, y, j)

                sample = np.sign(x)
                energy = model.evaluate(sample)
                tk += dtSB
                log.log(energy=energy, state=sample, positions=x)

            nb_operations = num_iterations * (2 * N**2 + 10 * N + 3)
            log.write_metadata(
                solution_state=sample, solution_energy=energy, total_operations=nb_operations
            )
        return sample, energy
