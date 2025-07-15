import numpy as np
import pathlib
import time
from numpy.random import MT19937, Generator
from collections.abc import Callable

# from ising.flow import LOGGER
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

    def set_params(self, resistance: float, capacitance: float, mu_param: float):
        """Set the parameters for the solver.

        Args:
            resistance (float): the resistance of the system.
            capacitance (float): the capacitance of the system.
            mu_param (float): the mu parameter for the ZIV diode.
        """
        self.resistance = resistance
        self.capacitance = capacitance
        self.mu_param = mu_param

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

    def mosfet(self, voltage:np.ndarray ):
        pass

    def dvdt(
        self,
        t: float,
        vt: np.ndarray,
        coupling: np.ndarray,
        frozen_nodes: np.ndarray | None = None,
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

        # Compute the voltage change dv
        dv = 1 / self.capacitance * (np.dot(coupling,  np.sign(vt)))

        # Ensure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt >= 1)
        cond2 = (dv < 0) & (vt <= -1)
        dv *= np.where(cond1 | cond2, 0.0, 1.0)

        if frozen_nodes is not None:
            dv[frozen_nodes] = 0.0

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
        frozen_nodes: np.ndarray | None = None,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
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
            file (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[np.ndarray, float]: the best energy and the best sample.
        """
        # Set up the time evaluations
        tend = dtMult * num_iterations

        self.set_params(resistance, capacitance, mu_param)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            self.bias = True
        else:
            new_model = model
            self.bias = False

        # Ensure the mean and variance of J are reasonable
        alpha = 1.
        J = alpha * triu_to_symm(new_model.J) * 1 / self.resistance


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
            while i < num_iterations or max_change <= stop_criterion:
                tk = i * dtMult
                k1 = dtMult * self.dvdt(tk, previous_voltages, J, frozen_nodes)
                k2 = dtMult * self.dvdt(tk + dtMult, previous_voltages + k1, J, frozen_nodes)
                k3 = dtMult * self.dvdt(tk + 1 / 2 * dtMult, previous_voltages + 1 / 4 * (k1 + k2), J, frozen_nodes)

                noise = self.noise(Temp, previous_voltages)
                new_voltages = previous_voltages + 1.0 / 6.0 * (k1 + k2 + 4.0 * k3) + noise
                Temp *= cooling_rate
                tk += dtMult
                i += 1

                # Log everything
                sample = np.sign(new_voltages[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

                # if i % 1000 == 0:
                #     LOGGER.info(f"Iteration {i} - time {tk:.2e} - energy {energy:.2f} - total flips {self.tot_sfs}")

                # Update the criterion changes
                if i > 0:
                    max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                        previous_voltages, ord=np.inf
                    )
                previous_voltages = np.copy(new_voltages)
            # LOGGER.info(f"Done at iteration {i} with energy {energy:.2f}")
            # Make sure to log to the last iteration if the stop criterion is reached
            # if max_change < stop_criterion:
            #     for j in range(i, num_iterations):
            #         tk += dtMult
            #         log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=tend)
        return sample, energy
