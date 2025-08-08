import numpy as np
import pathlib
from abc import abstractmethod
import copy

from ising.stages.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
# from ising.utils.clock import clock


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
        # return x, y

    def at(self, t, a0, dt, num_iterations):
        return a0 / (dt*num_iterations) * t

    def bt(self, t, a0, dt, num_iterations):
        return self.at(t, a0, dt, num_iterations) / 2

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
        bit_width:      int = 64
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
        floatmap      = {16:np.float16, 32:np.float32, 64:np.float64}
        dtype         = floatmap.get(bit_width, np.float16)
        J             = np.array(triu_to_symm(model.J), dtype=dtype)
        h             = np.array(model.h, dtype=dtype)
        initial_state = np.array(initial_state, dtype=dtype)
        x             = copy.deepcopy(initial_state)
        y             = 0.1 * initial_state

        # Cast all the variables to the correct data type
        dtSB = dtype(dtSB)
        a0   = dtype(a0)
        c0   = dtype(c0)
        tk   = dtype(0.0)

        schema = {
            "time"      : dtype,
            "energy"    : dtype,
            "state"     : (np.int8, (N,)),
            "positions" : (dtype, (N,)),
            "momenta"   : (dtype, (N,)),
            "at"        : dtype,
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
            log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=0.)
            for _ in range(num_iterations):
                atk = dtype(self.at(tk, a0, dtSB, num_iterations))
                # btk = dtype(self.bt(tk, a0, dtSB, num_iterations))

                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0 *  h) * dtSB
                x += self.update_x(y, dtSB, a0)

                for j in range(N):
                    if np.abs(x[j]) > 1:
                        self.update_rule(x, y, j)

                sample = np.sign(x)
                energy = model.evaluate(sample)
                tk    += dtSB
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk)

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
        # clocker = clock(clock_freq, clock_op)

        x = 0.01 * initial_state
        y = np.zeros_like(x)

        schema = {
            "time": float,
            "energy": np.float32,
            "state": (np.int8, (N,)),
            "positions": (np.float32, (N,)),
            "momenta": (np.float32, (N,)),
            "at": np.float32,
            # "time_clock": float,
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
            log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=0.)
            for _ in range(num_iterations):
                atk = self.at(tk, a0, dtSB, num_iterations)
                # btk = self.bt(tk, a0, dtSB, num_iterations)
                # clocker.add_operations(1)

                y += (-(a0 - atk) * x + c0 * np.matmul(J, np.sign(x)) + c0 * model.h) * dtSB
                # clocker.add_cycles(1 + np.log2(N))
                # clocker.add_operations(5 * N)
                # clocker.perform_operations()

                x += self.update_x(y, dtSB, a0)
                # clocker.add_operations(2 * N)
                # clocker.perform_operations()

                for j in range(N):
                    if np.abs(x[j]) > 1:
                        self.update_rule(x, y, j)
                        # clocker.add_operations(2)

                sample = np.sign(x)
                energy = model.evaluate(sample)

                # clocker.add_operations(1)
                # time_clock = clocker.perform_operations()
                tk += dtSB
                log.log(time=tk, energy=energy, state=sample, positions=x, momenta=y, at=atk)

            # total_time = clocker.get_time()
            nb_operations = num_iterations * (2 * N**2 + 10 * N + 3)
            log.write_metadata(
                solution_state=sample, solution_energy=energy, total_operations=nb_operations
            )
        return sample, energy
