import numpy as np
import pathlib
from abc import abstractmethod

from ising.flow import LOGGER
from ising.model.ising import IsingModel
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

    def at(self, t, a0, dt, num_iterations):
        return a0 / (dt*num_iterations) * t

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
        num_iterations: int,
        c0:             float,
        dtSB:           float,
        a0:             float = 1.0,
        file:           pathlib.Path | None = None,
        bit_width_x:      int = 16,
        bit_width_y:      int = 16,
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
        self.name += f"_{bit_width_x}_{bit_width_y}"
        N  = model.num_variables

        # Set up the model and initial states with the correct data type
        x_values        = np.linspace(-1, 1, 2**(bit_width_x)-1)
        y_values        = np.linspace(-1, 1, 2**(bit_width_y)-1)
        J             = np.array(triu_to_symm(model.J))
        h             = np.array(model.h)
        x             = np.zeros((model.num_variables,))
        y             = np.ones_like(x)

        schema = {
            "time"      : float,
            "energy"    : float,
            "state"     : (np.int8, (N,)),
            "positions" : (float, (N,)),
            "momenta"   : (float, (N,)),
            "at"        : float,
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
            tk = 0.0
            log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=0.)
            for i in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations)

                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0 *  h) * dtSB
                y = self.cast_to_values(y, y_values)
                x += self.update_x(y, dtSB, a0)
                x = self.cast_to_values(x, x_values)

                y = np.where(np.abs(x) >= 1, 0, y)
                x = np.where(np.abs(x) >= 1, np.sign(x), x)

                sample = np.sign(x)
                energy = model.evaluate(sample)
                tk    += dtSB
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk)

                if i % 1000 == 0:
                    LOGGER.info(f"Step {i} / {num_iterations} - Energy: {energy} -  max absolute voltage: {np.max(np.abs(x[:N]))}")

            nb_operations = num_iterations * (2 * N**2 + 10 * N + 3)
            log.write_metadata(
                solution_state=sample, solution_energy=energy, total_operations=nb_operations
            )
        return sample, energy