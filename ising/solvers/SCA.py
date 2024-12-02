import numpy as np
import pathlib
import random
import time

from ising.solvers.base import SolverBase
from ising.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger

class SCA(SolverBase):
    def change_hyperparam(self, param: float, rate: float) -> float:
        """Changes hyperparameters according to update rule."""
        return param * rate

    def solve(
        self,
        model: IsingModel,
        sample: np.ndarray,
        num_iterations: int,
        T: float,
        r_t: float,
        q: float,
        r_q: float,
        seed: int|None = None,
        file: pathlib.Path|None = None,
    ):
        """Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the
        [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper

        Args:
            model (IsingModel): instance of the Ising model that needs to be optimised.
            sample (np.ndarray): initial state of the Ising model.
            num_iterations (int): total amount of iterations which the solver needs to perform.
            T (float): temperature needed for the annealing process
            r_t (float): decrease rate of the temperature
            q (float): penalty parameter to ensure the copy states are equivalent to the real states.
            r_q (float): increase rate of the penalty parameter
            seed (int, None, optional): seed to generate random numbers. Important for reproducibility.
                                        Defaults to None.
            file (pathlib.Path, None, optional): absolute path to the logger file for logging the optimisation process.
                                                 If 'None', no logging is performed.

        Returns:
            sample, energy (tuple[np.ndarray, float]): final state and energy of the optimisation process.
        """
        N = model.num_variables
        hs = np.copy(model.h)
        flipped_states = []

        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        schema = {
            "energy": np.float32,
            "state": (np.int8, (N,))
        }

        metadata = {
            "solver": "Stochastic_cellular_automata_annealing",
            "initial_temp": T,
            "cooling_rate": r_t,
            "initial_penalty": q,
            "penalty_increase": r_q,
            "seed": seed,
            "num_iterations": num_iterations,
            "initial_state": sample
        }
        with HDF5Logger(file, schema) as log:
            log.write_metadata(metadata)

            for s in range(num_iterations):
                for x in range(N):
                    hs[x] += np.dot(model.J[x, :], sample)
                    P = self.get_prob(hs[x], sample[x], q, T)
                    rand = random.random()
                    if P < rand:
                        flipped_states.append(x)
                for x in flipped_states:
                    sample[x] = -sample[x]
                energy = model.evaluate(sample)
                log.log(energy=energy, state=sample)


                T = self.change_hyperparam(T, r_t)
                q = self.change_hyperparam(q, r_q)
                flipped_states = []

            log.write_metadata(solution_state=sample, solution_energy=energy)

        return sample, energy

    def get_prob(self, hsx:float, samplex:int, q:float, T:float)->float:
        """Calculates the probability of changing the value of a certain node
           according to SCA annealing process.

        Args:
            hsx (float): local field influence on the node.
            samplex (int): node
            q (float): penalty parameter
            T (float): temperature

        Returns:
            probability (float): probability of accepting the change.
        """
        val = hsx * samplex + q
        if -2 * T < val < 2 * T:
            return val / (4 * T) + 0.5
        elif val > 2 * T:
            return 1.0
        else:
            return 0.0
