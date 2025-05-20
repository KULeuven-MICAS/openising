import numpy as np
import pathlib
import time
from numpy.random import MT19937, Generator
from collections.abc import Callable

from ising.flow import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.helper_functions import return_rx


class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

    def do_spinflip(self, state: np.ndarray):
        # Generate random numbers
        rnd_v = self.rng_(state.shape)

        # Select nodes for spin flip
        self.chosen_nodes = rnd_v < self.p

        if self.bias:
            self.chosen_nodes[-1] = False

        # Apply spin flip
        absv = np.where(state > 0, -1, np.where(state < 0, 1, 0))
        self.flip_value = np.where(self.chosen_nodes, absv, 0.0)

        # Update statistics
        self.tot_sfs += np.sum(self.chosen_nodes)
        self.p += self.scale * self.sf_freq

    def set_params(self, resistance: float, capacitance: float, mu_param: float, flipping: bool):
        """Set the parameters for the solver.

        Args:
            resistance (float): the resistance of the system.
            capacitance (float): the capacitance of the system.
            mu_param (float): the mu parameter for the ZIV diode.
        """
        self.resistance = resistance
        self.capacitance = capacitance
        self.mu_param = mu_param
        self.flip_resistance = resistance / (1e4 * 128)
        self.flipping = flipping

    def set_spinflip(self, N: int, num_iterations: int, initial_prob: float, seed: int, flipping_freq: int):
        self.flip_value = np.zeros((N + int(self.bias),))
        self.chosen_nodes = np.full((N + int(self.bias),), False)
        self.tot_sfs = 0
        self.count = 0
        self.sh_iters = 10
        self.p0 = initial_prob
        self.p1 = 2e-7
        self.p = self.p0
        self.scale = ((self.p1 - self.p0) / float(num_iterations - 1)) if num_iterations > 1 else 0
        self.sf_freq = int(1 / flipping_freq)

        self.generator = Generator(MT19937(seed=seed))
        self.rng_: Callable = lambda size: self.generator.uniform(0, 1, size)

    def dvdt_flip(self, t: float, vt: np.ndarray, coupling: np.ndarray):
        """Differential equations for the multiplicative BRIM model.

        Args:
            t (float): time
            vt (np.ndarray): current voltages
            coupling (np.ndarray): coupling matrix J

        Returns:
            dv (np.ndarray): the change of the voltages
        """

        # set bias node to 1.
        if self.bias:
            vt[-1] = 1.0

        # Buffering
        c = 1 / np.abs(vt)

        # ZIV diode
        z = vt / self.resistance * (vt - 1) * (vt + 1) * self.mu_param

        # Flipping changes
        flip = np.where(self.chosen_nodes, (self.flip_value - vt), 0.0)

        # Compute the voltage change dv
        dv = 1 / self.capacitance * (np.dot(coupling, c * vt) - z + flip / self.flip_resistance)

        # Ensure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 1)
        cond2 = (dv < 0) & (vt < -1)
        dv *= np.where(cond1 | cond2, 0.0, 1.0)

        # Ensure the bias node does not change
        if self.bias:
            dv[-1] = 0.0
        return dv

    def dvdt(
        self,
        t: float,
        vt: np.ndarray,
        coupling: np.ndarray,
    ):
        """Differential equations for the multiplicative BRIM model when flipping is involved.

        Args:
            t (float): time
            vt (np.ndarray): current voltages
            coupling (np.ndarray): coupling matrix J

        Returns:
            dv (np.ndarray): the change of the voltages
        """

        # set bias node to 1.
        if self.bias:
            vt[-1] = 1.0

        # Buffering
        c = np.where(np.abs(vt) > 1., 1 / np.abs(vt), vt)

        # ZIV diode
        z = vt / self.resistance * (vt - 1) * (vt + 1) * self.mu_param

        # Compute the voltage change dv
        dv = 1 / self.capacitance * (np.dot(coupling, c * vt) - z)

        # Ensure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 1)
        cond2 = (dv < 0) & (vt < -1)
        dv *= np.where(cond1 | cond2, 0.0, 1.0)

        # Ensure the bias node does not change
        if self.bias:
            dv[-1] = 0.0
        return dv

    def noise(self, Temp: float, voltages: np.ndarray):
        if Temp != 0.0:
            noise = Temp * (np.random.normal(scale=1 / 1.96, size=voltages.shape))
            cond1 = (voltages >= 1) & (noise > 0)
            cond2 = (voltages <= -1) & (noise < 0)
            noise *= np.where(cond1 | cond2, 0.0, 1.0)
        else:
            noise = np.zeros_like(voltages)
        return noise

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        resistance: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        initial_temp_cont: float = 1.0,
        end_temp_cont: float = 0.05,
        stop_criterion: float = 1e-8,
        mu_param: float = -3.55,
        flipping: bool = False,
        flipping_freq: int = 10000,
        flipping_prob: float = 0.001799,
        flipping_time: float = 1e-3,
        file: pathlib.Path | None = None,
    ) -> tuple[float, np.ndarray]:
        """Solves the given problem using a multiplicative coupling scheme.

        Args:
            model (IsingModel): the model to solve.
            initial_state (np.ndarray): the initial spins of the nodes.
            dtMult (float): time step.
            num_iterations (int): the number of iterations.
            resistance (float, optional): the resisitance of the system. Defaults to 1.0.
            capacitance (float, optional): the capacitance of the system. Defaults to 1.0.
            seed (int, optional): the seed for random number generation. Defaults to 0.
            initial_temp_cont (float, optional): the initial temperature for the additive voltage noise.
                                                 Defaults to 1.0.
            end_temp_cont (float, optional): the final temperature for the additive voltage noise. Defaults to 0.05.
            stop_criterion (float, optional): the stopping criterion to stop the solver when the voltages don't change
                                              too much anymore. Defaults to 1e-8.
            coupling_annealing (bool, optional): whether to anneal the coupling matrix. Defaults to False.
            mu_param (float, optional): the mu parameter for the ZIV diode. Defaults to -3.55.
            flipping (bool, optional): whether to use the spin flip method. Defaults to False.
            flipping_freq (int, optional): the frequency of the spin flip method in Hertz. Defaults to 10 kHz.
            flipping_prob (float, optional): the probability of the spin flip method. Defaults to 0.001799.
            file (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[float, np.ndarray]: the best energy and the best sample.
        """
        # Set up the time evaluations
        tend = dtMult * num_iterations

        self.set_params(resistance, capacitance, mu_param, flipping)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            self.bias = True
        else:
            new_model = model
            self.bias = False
        J = triu_to_symm(new_model.J) * 2 / self.resistance

        # make sure the correct random seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the bias node and add noise to the initial voltages
        N = model.num_variables
        if self.bias:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state

        # Schema for logging
        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        # Set up the spin flipping
        self.set_spinflip(N, num_iterations, flipping_prob, seed, flipping_freq)
        flip_period = 1/(flipping_freq*dtMult)
        flip_iter = np.arange(0, num_iterations, flip_period)
        flip_iter = np.delete(flip_iter, 0)

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=np.sign(v[:-1]),
                model=model,
                num_iterations=num_iterations,
                time_step=dtMult,
                temperature=initial_temp_cont,
            )

            # Set up the simulation
            i = 0
            max_change = np.inf
            Temp = initial_temp_cont if initial_temp_cont < 1.0 else 0.5
            cooling_rate = (
                return_rx(num_iterations, initial_temp_cont, end_temp_cont) if initial_temp_cont != 0.0 else 1.0
            )
            previous_voltages = np.copy(v)
            energy = model.evaluate(np.sign(initial_state))

            log.log(time_clock=0.0, energy=energy, state=np.sign(initial_state), voltages=initial_state)

            while i < num_iterations and max_change > stop_criterion:
                tk = i * dtMult
                if flipping and i in flip_iter:
                    self.do_spinflip(previous_voltages)
                    dt_flip = dtMult * flipping_time
                    first_flip = np.argmax(self.chosen_nodes)
                    LOGGER.info(f"node {first_flip}, value: {previous_voltages[first_flip]}")
                    t_flip = tk
                    just_flipped = True
                    for j in range(int(flipping_time / dt_flip)):
                        k1 = self.dvdt_flip(tk, previous_voltages, J)
                        k2 = self.dvdt_flip(tk + dt_flip, previous_voltages + dt_flip * k1, J)
                        k3 = self.dvdt_flip(tk + 1 / 2 * dt_flip, previous_voltages + dt_flip / 4 * (k1 + k2), J)

                        # Add noise and update the voltages
                        noise = self.noise(Temp, previous_voltages)
                        new_voltages = previous_voltages + dt_flip / 6.0 * (k1 + k2 + 4.0 * k3) + noise
                        t_flip += dt_flip
                        if t_flip >= tk + dtMult:
                            LOGGER.info(f"node {first_flip}, value: {new_voltages[first_flip]}")
                            tk += dtMult
                            i += 1
                            Temp *= cooling_rate

                            energy = model.evaluate(np.sign(new_voltages[:N]))
                            log.log(
                                time_clock=tk, energy=energy, state=np.sign(new_voltages[:N]), voltages=new_voltages[:N]
                            )
                        previous_voltages = np.copy(new_voltages)

                    if t_flip < tk + dtMult:
                        dt = tk + dtMult - t_flip
                        i += 1
                        k1 = dt * self.dvdt(tk, previous_voltages, J)
                        k2 = dt * self.dvdt(tk + dtMult, previous_voltages + k1, J)
                        k3 = dt * self.dvdt(tk + 1 / 2 * dtMult, previous_voltages + 1 / 4 * (k1 + k2), J)

                        noise = self.noise(Temp, previous_voltages)
                        new_voltages = previous_voltages + 1.0 / 6.0 * (k1 + k2 + 4.0 * k3) + noise
                        Temp *= cooling_rate
                        tk += dtMult
                else:
                    just_flipped = False
                    k1 = dtMult * self.dvdt(tk, previous_voltages, J)
                    k2 = dtMult * self.dvdt(tk + dtMult, previous_voltages + k1, J)
                    k3 = dtMult * self.dvdt(tk + 1 / 2 * dtMult, previous_voltages + 1 / 4 * (k1 + k2), J)

                    noise = self.noise(Temp, previous_voltages)
                    new_voltages = previous_voltages + 1.0 / 6.0 * (k1 + k2 + 4.0 * k3) + noise
                    Temp *= cooling_rate
                    tk += dtMult
                    i += 1

                # Log everything
                sample = np.sign(new_voltages[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

                if i % 1000 == 0:
                    LOGGER.info(f"Iteration {i} - time {tk:.2e} - energy {energy:.2f} - total flips {self.tot_sfs}")

                # Update the criterion changes
                if i > 0 and not just_flipped:
                    max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                        previous_voltages, ord=np.inf
                    )
                previous_voltages = np.copy(new_voltages)
            LOGGER.info(f"Done at iteration {i} with energy {energy:.2f}")
            # Make sure to log to the last iteration if the stop criterion is reached
            if max_change < stop_criterion:
                for j in range(i, num_iterations):
                    tk += dtMult
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=tend)
        return sample, energy
