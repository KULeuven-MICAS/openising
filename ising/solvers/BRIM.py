from ising.solvers.solver import Solver
from ising.model.ising import IsingModel
import numpy as np
import pathlib


class BRIM(Solver):

    def k(self, kmax, kmin, cycle_duration, t):
        return kmax if int(t // (cycle_duration / 2)) % 2 == 0 else kmin

    def solve(
        self,
        model: IsingModel,
        file: pathlib.Path,
        v: np.ndarray,
        S: int,
        dt: float,
        kmin: float,
        kmax: float,
        C: float,
        G: float,
    ):
        N = model.num_variables
        tk = 0.0
        cycle_duration = dt * S / 10

        def dvdt(t, v):
            V = np.array([v] * N)
            k = self.k(kmin, kmax, cycle_duration, t)
            dv = (
                1 / C
                * (G * np.tanh(k * np.tanh(k * v)) - G * v - np.sum(model.J * (V - V.T), 0))
            )
            dv = np.where(np.all(np.array([dv > 0.0, v >= 1.0]), 0), np.zeros((N,)), dv)
            dv = np.where(np.all(np.array([dv < 0.0, v <= -1.0]), 0), np.zeros((N,)), dv)
            return dv

        with self.open_log(file, model, [f'v{i}' for i in range(N)]) as log:
            for i in range(self.S):
                k1 = dvdt(tk, v)
                k2 = dvdt(tk + dt / 2, v + dt / 2 * k1)
                k3 = dvdt(tk + dt / 2, v + dt / 2 * k2)
                k4 = dvdt(tk + dt, v + dt * k3)

                v += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                sample = self.set_sample(v)
                energy = model.evaluate(sample)
                tk += dt
                log.write(tk, energy, sample, v)
        return sample, energy
