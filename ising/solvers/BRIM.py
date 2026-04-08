import time
import numpy as np
import pathlib

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm

###
# This code is adapted from the code of the BRIM paper: https://ieeexplore.ieee.org/document/9407038/,
# given to us by professor Michael Huang.
###


class BRIM(SolverBase):
    """Implementation of the Bistable Resistively-coupled Ising Machine (BRIM) algorithm. The implementation is adapted\
          from the paper of [Afoakwa et al.](https://ieeexplore.ieee.org/document/9407038/)"""
    def __init__(self):
        super().__init__()
        self.name = "BRIM"
        self.end_prob = 2e-06
        self.sh_it_frac = 2e-04
        self.fourth = np.float32(1 / 4)
        self.three = np.float32(3)
        self.two_thirds = np.float32(2 / 3)

    def Ka(self, time: float, end_time: float) -> float:
        """Returns the coupling annealing term.

        @type time: float
        @param time: the time.
        @type end_time: float
        @param end_time: the end time.
        @rtype: float
        @return: Ka, the coupling annealing term.
        """
        return np.float32(1 - np.exp(-time / end_time))

    def choose_spinflips(self, voltages: np.ndarray):
        """ Chooses which spins to flip according to the flipping probability.
        @type voltages: np.ndarray
        @param voltages: the current voltages of the system.
        """
        # Increment count
        self.flip_counts = np.where(self.flip_counts != -1, self.flip_counts + 1, -1)

        # Choose which spins to flip
        random_v = np.random.uniform(0, 1, (self.num_variables + int(self.bias),))
        self.chosen_flips = random_v < self.p
        if self.bias:
            self.chosen_flips[-1] = False  # Do not flip the bias node
        self.flip_voltages = np.where(self.chosen_flips, -np.abs(voltages), 0).astype(np.float32)

        # Reset counter for chosen flips
        self.flip_counts = np.where(self.chosen_flips, 0, self.flip_counts)

        # Unclamp nodes with spin flip timeout
        time_up = self.flip_counts == self.flip_iters
        self.flip_voltages = np.where(time_up, 0, self.flip_voltages)
        self.chosen_flips = np.where(time_up, False, self.chosen_flips)
        self.flip_counts = np.where(time_up, -1, self.flip_counts)

        self.p += self.prob_change

    def dvdt(self, t, vt, coupling, Ka) -> np.ndarray:
        """Computes the derivative or change of the voltages at time t.

        @type t: float
        @param t: the time.
        @type vt: np.ndarray
        @param vt: the current voltages of the system.
        @type coupling: np.ndarray
        @param coupling: the coupling matrix of the system.
        @type Ka: float
        @param Ka: the coupling annealing term.
        @rtype: np.ndarray
        @return: the change of the voltages at time t.
        """
        # Make sure the bias node is 1
        if self.bias:
            vt[-1] = 1.0

        # ZIV diode
        z = vt / self.resistance + (
            (-2.156334025305975e-05 * np.power(vt, 5))
            + (1.017179575405042e-04 * np.power(vt, 3))
            + (-2.231312342175098e-05 * vt)
        )

        # Compute the differential equation
        dv = np.zeros_like(vt, dtype=np.float32)
        if self.do_flipping:
            flip = (self.flip_voltages - vt) / self.resistance
            dv = np.where(self.chosen_flips, flip / self.capacitance, 1 / self.capacitance * ((Ka * coupling) @ vt - z))
        else:
            dv = 1 / self.capacitance * ((Ka * coupling) @ vt - z)

        # Make sure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 1)
        cond2 = (dv < 0) & (vt < -1)
        dv *= np.where(cond1 | cond2, 0.0, 1)

        # Make sure the bias node does not change
        if self.bias:
            dv[-1] = 0.0
        return dv

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        dtBRIM: float,
        capacitance: float,
        resistance: float,
        probability_start: float = 0.001799,
        stop_criterion: float = 1e-8,
        file: pathlib.Path | None = None,
        coupling_annealing: bool = False,
        do_flipping: bool = False,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Simulates the BRIM dynamics by integrating the Lyapunov equation through time with the RK4 method.

        @type model: IsingModel
        @param model: the model of which the optimum needs to be found.
        @type initial_state: np.ndarray
        @param initial_state: initial spins of the nodes.
        @type num_iterations: int
        @param num_iterations: amount of iterations that need to be simulated
        @type dtBRIM: float
        @param dtBRIM: time step.
        @type capacitance: float
        @param capacitance: capacitor of system.
        @type probability_start: float, optional
        @type resistance: float
        @param resistance: resistance of the system.
        @type probability_start: float, optional
        @param probability_start: initial flipping probability. Defaults to 0.001799.
        @type stop_criterion: float, optional
        @param stop_criterion: stop criterion for the maximum allowed change of voltages between iterations.\
                                              Defaults to 1e-8.
        @type file: pathlib.Path, None, Optional
        @param file: absolute path to which data will be logged. If 'None',\
                                                 nothing is logged.
        @type coupling_annealing: bool, optional
        @param coupling_annealing: whether to anneal the coupling matrix. Defaults to False.
        @type do_flipping: bool, optional
        @param do_flipping: whether to perform spin flips. Defaults to False.
        @type seed: int, optional
        @param seed: seed for the random number generator. Defaults to 0.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: optimal state, optimal energy, total simulation time, amount of operations, performed iterations.
        """

        # Set the time evaluations
        dtBRIM = np.float32(dtBRIM)
        tend = dtBRIM * num_iterations
        t_eval = np.linspace(0.0, tend, num_iterations, dtype=np.float32)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            self.bias = True
        else:
            new_model = model
            self.bias = False
        coupling = triu_to_symm(new_model.J).astype(np.float32) / resistance

        res_average = np.mean(np.sum(coupling, axis=1))
        tau = capacitance/res_average

        # Make sure the correct seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Ensure the bias node is added and add noise to the initial voltages
        if self.bias:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state
        v = v.astype(np.float32)
        # Schema for the logging
        schema = {
            "time": float,
            "energy": np.float32,
        }

        # Set up the flipping probability
        if probability_start <= self.end_prob:
            self.end_prob = probability_start
        self.prob_change = (self.end_prob - probability_start) / (num_iterations - 1)

        assert self.prob_change <= 0.0, "probability change should be negative"
        self.p = probability_start
        # Set all parameters
        self.num_variables = model.num_variables
        self.capacitance = np.float32(capacitance)
        self.resistance = np.float32(resistance)
        self.do_flipping = do_flipping
        self.flip_iters = num_iterations * self.sh_it_frac
        self.chosen_flips = np.zeros(self.num_variables + int(self.bias), dtype=bool)
        self.flip_voltages = np.zeros(self.num_variables + int(self.bias), dtype=np.float32)
        self.flip_counts = -np.ones(self.num_variables + int(self.bias), dtype=np.int32)

        # operation count param
        time_zero = 0
        nb_operations = 0

        with HDF5Logger(file, schema) as log:
            # Log the initial metadata
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(v),
                    model=model,
                    num_iterations=num_iterations,
                    C=capacitance,
                    time_step=dtBRIM,
                    seed=seed,
                    stop_criterion=stop_criterion,
                    coupling_annealing=coupling_annealing,
                )

            # Initialize the simulation variables
            i = 0
            previous_voltages = np.copy(v).astype(np.float32)
            max_change = np.inf

            # Initial logging
            if log.filename is not None:
                sample = np.sign(v[: model.num_variables])
                energy = model.evaluate(sample)
                log.log(time=0.0, energy=energy)

            while i < (num_iterations) and max_change > stop_criterion:
                tk = t_eval[i]

                if coupling_annealing:
                    Ka = self.Ka(tk, tend)
                else:
                    Ka = np.float32(1.0)

                # Forward Euler
                k1 = dtBRIM * self.dvdt(tk, previous_voltages, coupling, Ka)

                new_voltages = np.clip(previous_voltages + k1, np.float32(-1), np.float32(1))

                if self.do_flipping:
                    self.choose_spinflips(new_voltages)
                # Log everything
                if log.filename is not None:
                    sample = np.sign(new_voltages[: model.num_variables])
                    energy = model.evaluate(sample)
                    log.log(time=tk, energy=energy)
                if tk - time_zero >= tau:
                    time_zero = tk
                    nb_operations += 2 * model.num_variables**2 + 3 * model.num_variables

                # Update criterion changes
                if i > 0:
                    max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                        previous_voltages, ord=np.inf
                    )
                previous_voltages = np.copy(new_voltages)
                i += 1

            # Make sure to log to the last iteration if the stop criterion is reached
            if max_change < stop_criterion:
                for j in range(i, num_iterations):
                    tk = t_eval[j]
                    if log.filename is not None:
                        log.log(time=tk, energy=energy)
                    if tk - time_zero >= tau:
                        time_zero = tk
                        nb_operations += 2 * model.num_variables**2 + 3 * model.num_variables
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample, solution_energy=energy, total_time=t_eval[-1], total_operations=nb_operations
                )
            else:
                sample = np.sign(new_voltages[: model.num_variables])
                energy = model.evaluate(np.sign(new_voltages[: model.num_variables]))
        return sample, energy, tend, nb_operations, -1
