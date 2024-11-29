from ising.model.ising import IsingModel
from ising.solvers.solver import Solver
import numpy as np
import pathlib


class SB(Solver):
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

    def solve(self, model: IsingModel):
        pass


class ballisticSB(SB):
    def solve(
        self,
        model: IsingModel,
        file: pathlib.Path,
        x: np.ndarray,
        y: np.ndarray,
        S: int,
        a0: float,
        at: callable,
        c0: float,
        dt: float,
    ):
        tk = 0.0
        N = model.num_variables
        with self.open_log(file, model, [f"x{i}" for i in range(N)]) as log:
            for i in range(S):
                for j in range(N):
                    y[j] += (-(a0 - at(tk)) * x[j] + c0 * np.dot(model.J[:, j], x) + c0 * model.h[j]) * dt
                    x[j] += self.update_x(y, dt, a0, j)
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                sample = np.sign(x)
                energy = model.evaluate(sample)
                log.write(tk, energy, sample, x)
                tk += dt
        return sample, energy


class discreteSB(SB):
    def solve(
        self,
        model: IsingModel,
        file: pathlib.Path,
        x: np.ndarray,
        y: np.ndarray,
        S: int,
        a0: float,
        at: callable,
        c0: float,
        dt: float,
    ):
        N = model.num_variables
        tk = 0.0
        with self.open_log(file, model, [f'x{i}' for i in range(N)]) as log:
            for i in range(S):
                for j in range(N):
                    y[j] += (
                        -(a0 - at(tk)) * x[j]
                        + c0 * np.inner(model.J[:, j], np.sign(x))
                        + c0 * model.h[j]
                    ) * dt
                    x[j] += self.update_x(y, dt, a0, j)
                    if np.abs(x[j]) > 1:
                        x, y = self.update_rule(x, y, j)
                sample = np.sign()  # TODO: I think this is broken
                energy = model.evaluate(sample)
                tk += dt
                log.write(tk, energy, sample, x)
        return sample, energy
