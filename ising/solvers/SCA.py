import numpy as np
import pathlib
import random
import time

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.clock import clock


class SCA(SolverBase):
    def __init__(self):
        self.name = "SCA"

    def change_hyperparam(self, param: float, rate: float) -> float:
        """Changes hyperparameters according to update rule."""
        return param * rate

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp: float,
        cooling_rate: float,
        q: float,
        r_q: float,
        seed: int | None = None,
        file: pathlib.Path | None = None,
        clock_freq: float = 1e6,
        clock_op: int = 1000,
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
        hs = np.zeros((N,))
        J = triu_to_symm(model.J)
        flipped_states = []
        clocker = clock(clock_freq, clock_op)
        initial_state = np.sign(initial_state)

        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        schema = {"energy": np.float32, "state": (np.int8, (N,)), "time_clock": float}

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=initial_state,
                model=model,
                num_iterations=num_iterations,
                initial_temp=initial_temp,
                cooling_rate=cooling_rate,
                initial_penalty=q,
                penalty_increase=r_q,
                seed=seed,
            )
            T = initial_temp
            for _ in range(num_iterations):
                hs = np.matmul(J, initial_state) + model.h
                clocker.add_cycles(1 + np.log2(N))

                Prob = self.get_prob(hs, initial_state, q, T)
                clocker.add_operations(5 * N)
                rand = np.random.rand(N)
                clocker.add_operations(N)
                clocker.perform_operations()

                flipped_states = [y for y in range(N) if Prob[y] < rand[y]]
                clocker.add_operations(N)

                initial_state[flipped_states] = -initial_state[flipped_states]
                energy = model.evaluate(initial_state)
                clocker.add_operations(4)
                time_clock = clocker.perform_operations()

                log.log(energy=energy, state=initial_state, time_clock=time_clock)

                T = self.change_hyperparam(T, cooling_rate)
                q = self.change_hyperparam(q, r_q)
                flipped_states = []

            total_time = clocker.get_time()
            nb_operations = num_iterations * (2 * N**2 + 8 * N + N / 2 + 2)
            log.write_metadata(
                solution_state=initial_state,
                solution_energy=energy,
                total_time=total_time,
                total_operations=nb_operations,
            )

        return initial_state, energy

    def get_prob(self, hs: np.ndarray, sample: np.ndarray, q: float, T: float) -> np.ndarray:
        """Calculates the probability of changing the value of the spins
           according to SCA annealing process.

        Args:
            hs (np.ndarray): local field influence.
            sample (np.ndarray): spin of the nodes.
            q (float): penalty parameter
            T (float): temperature

        Returns:
            probability (np.ndarray): probability of accepting the change of all nodes.
        """
        val = 1 / T * (np.multiply(hs, sample) + q) / 2
        return 1 / (1 + np.exp(-val))
