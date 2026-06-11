import numpy as np
import pathlib
import time
from abc import abstractmethod

from ising.stages.model.ising import IsingModel
from ising.solvers.base import SolverBase
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.flow import return_c0


class SB(SolverBase):
    """Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    This class inherits from the abstract Solver base class.
    """

    def __init__(self):
        super().__init__()
        self.name = "SB"
        self.max_energy_change = 1e-6

    def update_x(self, y, dt, a0):
        return a0 * y * dt

    def update_rule(self, x, y, node):
        x[node] = np.sign(x[node])
        y[node] = 0.0

    def at(self, t, a0, dt, num_iterations) -> float:
        return a0 / (dt * num_iterations) * t

    @abstractmethod
    def solve(self, model: IsingModel):
        pass


class ballisticSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"b{self.name}"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtbSB: float,
        c0: float = 0.0,
        a0: float = 1.0,
        seed: int = 0,
        file: pathlib.Path | None = None,
        stop_criterion: bool = False,
    ) -> tuple[np.ndarray, float, float, int, int]:
        """Performs the ballistic Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation introduces perfectly inelastic walls at |x_i| = 1
        to reduce analog errors.

        @type model: IsingModel
        @param model: the model of which the optimum needs to be found.
        @type initial_state: np.ndarray
        @param initial_state: initial discrete state of the system.
        @type num_iterations: int
        @param num_iterations: amount of iterations that needs to be performed.
        @type dtSB: float
        @param dtSB: time step of the system.
        @type c0: float, optional
        @param c0: Ising energy contribution to the Hamiltonian. Defaults to 0.0, which corresponds\
              to the optimal value.
        @type a0: float, optional
        @param a0: value to which the bifurcation parameter will converge to. Defaults to 1.
        @type seed: int, optional
        @param seed: random seed for the  initialization. Defaults to 0 which means a random seed\
            will be used.
        @type file: pathlib.Path, None, optional
        @param file: full path to which data will be logged. If 'None', \
            no logging is performed.
        @type stop_criterion: bool, optional
        @param stop_criterion: whether to stop the algorithm on stagnation of the energy or not.\
                                             Defaults to False.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: optimal solution state, optimal solution energy, total CPU time to perform the algorithm,\
              amount of operations, amount of performed iterations
        """
        N = model.num_variables

        if not stop_criterion:
            self.zero_en_length = num_iterations
        if c0 == 0.0:
            c0 = return_c0(model)
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the model and initial states with the correct data type
        J = np.array(triu_to_symm(model.J), dtype=np.float32)
        h = np.array(model.h)
        initial_state = np.array(initial_state)
        x = np.zeros_like(initial_state, dtype=np.float32)
        y = np.random.uniform(-0.1, 0.1, (model.num_variables,)).astype(np.float32)

        schema = {
            "time": np.float32,
            "energy": float,
            "positions": (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(x),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtbSB,
                    a0=a0,
                    c0=c0,
                )

            tk = 0.0
            sample = np.sign(x)
            energy = model.evaluate(sample)
            if log.filename is not None:
                log.log(time=0.0, energy=energy, positions=x)
            k = 0
            current_length = 0
            start_time = time.time()
            while k < num_iterations and current_length < self.zero_en_length:
                atk = self.at(tk, a0, dtbSB, num_iterations)  # 4

                y += (-(a0 - atk) * x + c0 * np.matmul(J, x) + c0 * h) * dtbSB
                # 1 + N + 2*N**2 + N + N + 2*N + N= 2*N**2 + 6*N + 1
                x += self.update_x(y, dtbSB, a0)  # N+1

                y = np.where(np.abs(x) >= 1, 0, y)  # N
                x = np.where(np.abs(x) >= 1, np.sign(x), x)  # N

                tk += dtbSB  # 1
                k += 1
                sample = np.sign(x)
                if stop_criterion or log.filename is not None:
                    energy_new = model.evaluate(sample)
                if log.filename is not None:
                    elapsed_time = time.time() - start_time
                    log.log(time=elapsed_time, energy=energy_new, positions=x)
                if stop_criterion:
                    current_length += int(
                        self.handle_stop_criterion(energy, energy_new) < self.max_energy_change and stop_criterion
                    )
                    energy = energy_new

            nb_operations = num_iterations * (2 * N**2 + 9 * N + 6)
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_operations=nb_operations,
                    total_time=elapsed_time,
                    total_iterations=k,
                )
            else:
                sample = np.sign(x)
                energy = model.evaluate(sample)
                elapsed_time = time.time() - start_time
        return sample, energy, elapsed_time, nb_operations, k


class discreteSB(SB):
    def __init__(self):
        super().__init__()
        self.name = f"d{self.name}"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtdSB: float,
        c0: float = 0.0,
        a0: float = 1.0,
        seed: int = 0,
        file: pathlib.Path | None = None,
        stop_criterion: bool = False,
    ) -> tuple[np.ndarray, float, float, int, int]:
        """Performs the discrete Simulated Bifurcation algorithm first proposed by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953).
        This variation of Simulated Bifurcation discretizes the positions x_i at all times to reduce analog errors.

        @type model: IsingModel
        @param model: the model of which the optimum needs to be found.
        @type initial_state: np.ndarray
        @param initial_state: initial discrete state of the system.
        @type num_iterations: int
        @param num_iterations: amount of iterations that needs to be performed.
        @type dtSB: float
        @param dtSB: time step of the system.
        @type c0: float, optional
        @param c0: Ising energy contribution to the Hamiltonian. Defaults to 0.0, which corresponds\
              to the optimal value.
        @type a0: float, optional
        @param a0: value to which the bifurcation parameter will converge to. Defaults to 1.
        @type seed: int, optional
        @param seed: random seed for the  initialization. Defaults to 0 which means a random seed\
            will be used.
        @type file: pathlib.Path, None, optional
        @param file: full path to which data will be logged. If 'None', \
            no logging is performed.
        @type stop_criterion: bool, optional
        @param stop_criterion: whether to stop the algorithm on stagnation of the energy or not.\
                                             Defaults to False.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: optimal solution state, optimal solution energy, total CPU time to perform the algorithm,\
              amount of operations, amount of performed iterations
        """
        N = model.num_variables
        tk = 0.0
        if c0 == 0.0:
            c0 = return_c0(model)

        if not stop_criterion:
            self.zero_en_length = num_iterations
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the model and initial states with the correct data type
        J = np.array(triu_to_symm(model.J), dtype=np.float32)
        h = np.array(model.h)
        initial_state = np.array(initial_state)
        x = np.zeros_like(initial_state, dtype=np.float32)
        y = np.random.uniform(-0.1, 0.1, (model.num_variables,)).astype(np.float32)

        schema = {
            "time": np.float32,
            "energy": np.float32,
            "positions": (np.float32, (N,)),
        }

        with HDF5Logger(file, schema) as log:
            sample = np.sign(x)
            energy = model.evaluate(sample)
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(x),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtdSB,
                    a0=a0,
                    c0=c0,
                )
                log.log(time=0.0, energy=energy, positions=x)
            k = 0
            energy_old = energy
            current_length = 0
            start_time = time.time()
            while k < num_iterations and current_length < self.zero_en_length:
                atk = self.at(tk, a0, dtdSB, num_iterations)  # 3

                y += (-(a0 - atk) * x + c0 * np.matmul(J, np.sign(x)) + c0 * h) * dtdSB
                # 1+N + 2*N**2 + N + N + 2*N + N = 2*N**2 + 6*N + 1
                x += self.update_x(y, dtdSB, a0)  # N+1

                y = np.where(np.abs(x) >= 1, 0, y)  # N
                x = np.where(np.abs(x) >= 1, np.sign(x), x)  # N

                tk += dtdSB  # 1
                if log.filename is not None:
                    elapsed_time = time.time() - start_time
                    sample = np.sign(x)
                    energy = model.evaluate(sample)
                    log.log(time=elapsed_time, energy=energy, positions=x)
                k += 1
                current_length += int(
                    self.handle_stop_criterion(energy_old, energy) < self.max_energy_change and stop_criterion
                )
                energy_old = energy
            nb_operations = num_iterations * (2 * N**2 + 9 * N + 5)
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_operations=nb_operations,
                    total_time=elapsed_time,
                    total_iterations=k,
                )
            else:
                elapsed_time = time.time() - start_time
                sample = np.sign(x)
                energy = model.evaluate(np.sign(x))
        return sample, energy, elapsed_time, nb_operations, k
