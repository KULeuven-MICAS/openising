import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class BRIM(SolverBase):

    def __init__(self):
        self.name = 'BRIM'

    def k(self, kmax: float, kmin: float, t: float, t_final: float) -> float:
        """Returns the gain of the latches at time t.

        Args:
            kmax (float): maximum gain of the latch
            kmin (float): minimum gain of the latch
            tend (float): end time of the simulation
            t (float): time

        Returns:
            float: latch gain
        """
        return kmin + ((kmax - kmin) / t_final) * t

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
        random_flip: bool = False,
        seed: int = 1,
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
        t_eval = np.linspace(0.0, tend, num_iterations)
        new_model = model.transform_to_no_h()
        J = triu_to_symm(new_model.J)
        v = np.block([v, 1.])
        flip_it = t_eval[:100]
        np.random.seed(seed)
        v += 0.01 * (np.random.random((N+1,)) - 0.5)

        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        def dvdt(t, vt):
            if random_flip and t in flip_it:
                flip = np.random.choice(N)
                vt[flip] = -vt[flip]
            vt[-1] = 1.0
            V = np.array([vt] * (N+1))
            k = self.k(kmax, kmin, t=t, t_final=tend)
            dv = 1 / C * (G * np.tanh(k * np.tanh(k * vt)) - G * vt - np.sum(J * (V - V.T), 0))
            dv = np.where(np.all(np.array([dv > 0.0, vt >= 1.0]), 0), np.zeros((N+1,)), dv)
            dv = np.where(np.all(np.array([dv < 0.0, vt <= -1.0]), 0), np.zeros((N+1,)), dv)
            dv[-1] = 0.
            return dv

        with HDF5Logger(file, schema) as log:
            self.log_metadata(logger=log,
                              initial_state=np.sign(v),
                              model=model,
                              num_iterations=num_iterations,
                              G=G,
                              C=C,
                              time_step=dt,
                              random_flip=random_flip,
                              seed=seed,
                              kmax=kmax,
                              kmin=kmin)

            vi = np.copy(v)
            for i in range(num_iterations):
                time = t_eval[i]
                k1 = dt * dvdt(time, vi)
                k2 = dt * dvdt(time + 0.5*dt, vi + 0.5*k1)
                k3 = dt * dvdt(time + 0.5 * dt, vi + 0.5 * k2)
                k4 = dt * dvdt(time + dt, vi + k3)

                vi += 1./6. * (k1 + 2*k2 + 2*k3 + k4)
                sample = np.sign(vi[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=time, energy=energy, state=sample, voltages=vi[:N])

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy
