import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        file: pathlib.Path|None=None,
    ) -> tuple[float, np.ndarray]:
        """Solves the given problem using a multiplicative coupling scheme.

        Args:
            model (IsingModel): the model to solve.
            v (np.ndarray): the initial voltages.
            dt (float): time step.
            num_iterations (int): the number of iterations.
            logfile (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: the best energy and the best sample.
        """
        # print(f"{dt=}")
        N = model.num_variables
        tend = dtMult * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations)

        new_model = model.transform_to_no_h()
        J = triu_to_symm(new_model.J)
        v = np.block([0.5*initial_state, 1.0])

        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        def dvdt(t, vt):
            vt[-1] = 1.0
            k = np.tanh(3*vt)
            coupling = 1 / 2 * np.dot(J, k)
            cond1 = (coupling > 0) & (vt > 0)
            cond2 = (coupling < 0) & (vt < 0)
            dv = coupling * np.where(cond1 | cond2, 1 - v**2, 1)
            # print(f"{dv=}")
            dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log, initial_state=np.sign(v[:-1]), model=model, num_iterations=num_iterations, time_step=dtMult
            )
            for i in range(num_iterations):
                tk = t_eval[i]
                k1 = dtMult * dvdt(tk, v)
                k2 = dtMult * dvdt(tk + 2 / 3 * dtMult, v + 2 / 3 * k1)
                # print(f"{k2=}")
                v += 1.0 / 4.0 * (k1 + 3.0 * k2)
                sample = np.sign(v[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=v[:N])
            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy
