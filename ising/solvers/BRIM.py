import numpy as np
import pathlib
from scipy.integrate import solve_ivp

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class BRIM(SolverBase):
    def k(self, kmax: float, kmin: float, t: float, tend: float) -> float:
        """Returns the gain of the latches at time t.

        Args:
            kmax (float): maximum gain of the latch
            kmin (float): minimum gain of the latch
            tend (float): end time of the simulation
            t (float): _description_

        Returns:
            float: latch gain
        """
        return kmin + (kmax - kmin) / tend * t

    def solve(
        self,
        model: IsingModel,
        v: np.ndarray,
        num_iterations: int,
        dt: float,
        kmin: float,
        kmax: float,
        C: float,
        G: float,
        file: pathlib.Path | None = None,
        random_flip:bool=False,
        seed:int=1
    ) -> tuple[np.ndarray, float]:
        """Simulates the BLIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

        Args:
            model (IsingModel): the model of which the optimum needs to be found.
            v (np.ndarray): initial voltages of the nodes
            num_iterations (int): amount of iterations that need to be simulated
            dt (float): time step.
            kmin (float): minimal gain of the latch
            kmax (float): maximum gain of the latch
            C (float): capacitor parameter.
            G (float): resistance parameter.
            file (pathlib.Path, None, Optional): absolute path to which data will be logged. If 'None',
                                                 nothing is logged.

        Returns:
            (sample, energy) tuple[np.ndarray, float]: optimal sample and energy.
        """
        N = model.num_variables
        tend = dt * num_iterations
        J = triu_to_symm(model.J)
        np.random.seed(seed)

        schema = {"time": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        metadata = {
            "solver": "BLIM",
            "initial_state": np.sign(v),
            "num_iterations": num_iterations,
            "time_step": dt,
            "C": C,
            "G": G,
            "kmin": kmin,
            "kmax": kmax,
            "random_flip": random_flip,
            "seed": seed
        }

        def dvdt(t, v):
            V = np.array([v] * N)
            k = self.k(kmax, kmin, t, tend)
            dv = 1 / C * (G * np.tanh(k * np.tanh(k * v)) - G * v - np.sum(J * (V - V.T), 0))

            dv = np.where(np.all(np.array([dv > 0.0, v >= 1.0]), 0), np.zeros((N,)), dv)
            dv = np.where(np.all(np.array([dv < 0.0, v <= -1.0]), 0), np.zeros((N,)), dv)
            if random_flip and t % (100 * dt) == 0:
                flip = np.random.choice(N)
                dv[flip] = -2.*v[flip]
            return dv

        with HDF5Logger(file, schema) as log:
            log.write_metadata(**metadata)
            t_eval = np.linspace(0., tend, num_iterations)
            sol = solve_ivp(fun=dvdt, t_span=(0.0, tend), y0=v, t_eval=t_eval)
            for t, vi in zip(sol.t, sol.y.T):
                sample = np.sign(vi)
                energy = model.evaluate(sample)
                log.log(time=t, energy=energy, state=sample, voltages=vi)

            # for i in range(num_iterations):
            #     k1 = dvdt(tk, v)
            #     k2 = dvdt(tk + dt / 2, v + dt / 2 * k1)
            #     k3 = dvdt(tk + dt / 2, v + dt / 2 * k2)
            #     k4 = dvdt(tk + dt, v + dt * k3)

            #     v += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            #     sample = np.sign(v)
            #     energy = model.evaluate(sample)
            #     log.log(time=tk, energy=energy, state=sample, voltages=v)
            #     tk += dt
            log.write_metadata(solution_state=sample, solution_energy=energy)
        return sample, energy
