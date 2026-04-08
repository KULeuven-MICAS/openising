import numpy as np
import pathlib
import random
import time

from ising.stages import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.flow import return_q


class SCA(SolverBase):
    def __init__(self):
        super().__init__()
        self.name = "SCA"

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        initial_temp_SCA: float,
        cooling_rate_SCA: float,
        q: float,
        r_q: float,
        seed: int | None = None,
        stop_criterion: bool = False,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float, float, int, int]:
        """Implementation of the Stochastic Cellular Automata (SCA) annealing algorithm of the
        [STATICA](https://ieeexplore.ieee.org/document/9222223/?arnumber=9222223) paper

        @type model: IsingModel
        @param model: instance of the Ising model that needs to be optimised.
        @type initial_state: np.ndarray
        @param initial_state: initial state of the Ising model.
        @type num_iterations: int
        @param num_iterations: total amount of iterations which the solver needs to perform.
        @type initial_temp_SCA: float
        @param initial_temp_SCA: temperature needed for the annealing process
        @type cooling_rate_SCA: float
        @param cooling_rate_SCA: decrease rate of the temperature.
        @type q: float
        @param q: penalty parameter to ensure the copy states are equivalent to the real states.
        @type r_q: float
        @param r_q: increase rate of the penalty parameter
        @type seed: int, None, optional
        @param seed: seed to generate random numbers. Important for reproducibility.\
                                        Defaults to None.
        @type stop_criterion: bool, optional
        @param stop_criterion: whether to stop the solver on convergence of the energy. Defaults to False.
        @type file: pathlib.Path, None, optional
        @param file: absolute path to the logger file for logging the optimisation process.\
                                                 If 'None', no logging is performed.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: optimal state, optimal energy, total simulation time, amount of operations, performed iterations.
        """
        if q == -1.0:
            q = return_q(model)
            LOGGER.info(f"Using optimal q value: {q}")
            r_q = 1.0

        if not stop_criterion:
            self.zero_en_length = num_iterations
        N = model.num_variables
        hs = np.zeros((N,))
        J = triu_to_symm(model.J)
        flipped_states = []
        state = np.copy(np.sign(initial_state))
        if seed is None:
            seed = int(time.time() * 1000)
        random.seed(seed)

        schema = {"time": np.float32, "energy": np.float32, "state": (np.int8, (N,))}

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=state,
                    model=model,
                    num_iterations=num_iterations,
                    initial_temp=initial_temp_SCA,
                    cooling_rate=cooling_rate_SCA,
                    initial_penalty=q,
                    penalty_increase=r_q,
                    seed=seed,
                )
            k = 0
            current_length = 0
            start_time = time.time()
            T = initial_temp_SCA
            energy = model.evaluate(state.astype(np.float32))
            if log.filename is not None:
                log.log(time=0.0, energy=energy, state=state)
            while k < num_iterations and current_length < self.zero_en_length:
                hs = J @ state + model.h  # 2*N**2 + N

                Prob = self.get_prob(hs, state, q, T)  # 2*N + 3*N
                rand = np.random.uniform(0, 1, size=(N,))  # N

                flipped_states = [y for y in range(N) if Prob[y] < rand[y]]  # N

                state[flipped_states] = -state[flipped_states]  # N

                T = T * cooling_rate_SCA  # 1
                q = q * r_q  # 1
                flipped_states = []
                energy_new = model.evaluate(state.astype(np.float32))
                if log.filename is not None:
                    elapsed_time = time.time() - start_time
                    log.log(time=elapsed_time, energy=energy_new, state=state)

                current_length += int(
                    self.handle_stop_criterion(energy, energy_new) < self.max_energy_change and stop_criterion
                )
                energy = energy_new
                k += 1

            nb_operations = num_iterations * (2 * N**2 + 8 * N + N / 2 + 2)
            if log.filename is not None:
                log.write_metadata(
                    total_time=elapsed_time,
                    solution_state=state,
                    solution_energy=energy,
                    total_operations=nb_operations,
                )
            else:
                elapsed_time = time.time() - start_time
                energy = model.evaluate(state.astype(np.float32))

        return state, energy, elapsed_time, nb_operations, k

    def get_prob(self, hs: np.ndarray, sample: np.ndarray, q: float, T: float) -> np.ndarray:
        """
        Calculates the probability of changing the value of the spins according to SCA annealing process.

        @type hs: np.ndarray
        @param hs : local field influence.
        @type sample: np.ndarray
        @param sample: spin of the nodes.
        @type q: float
        @param q: penalty parameter
        @type T: float
        @param T: temperature
        @rtype: np.ndarray
        @return: probability of accepting the change of all nodes.

        """
        val = hs * sample + q
        probs = np.zeros_like(val)
        for i, value in enumerate(val):
            if value >= -2 * T and value <= 2 * T:
                probs[i] = value / (4 * T) + 0.5
            elif value > 2 * T:
                probs[i] = 1
            else:
                probs[i] = 0
        return probs
