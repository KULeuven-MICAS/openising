import numpy as np
import pathlib
import time

from ising.stages import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm
from ising.utils.helper_functions import return_rx
from ising.utils.numba_functions import dvdt_solver

class Multiplicative(SolverBase):
    def __init__(self):
        self.name = "Multiplicative"

    def set_params(
        self,
        dt: float,
        num_iterations: int,
        resistance: float,
        capacitance: float,
        stop_criterion: float,
        initial_temp_cont: float,
        end_temp_cont: float,
        coupling: np.ndarray,
    ):
        """Set the parameters for the solver.

        Args:
            resistance (float): the resistance of the system.
            capacitance (float): the capacitance of the system.
            frozen_nodes (np.ndarray | None): the nodes that are frozen.
            stop_criterion (float): the stopping criterion to stop the solver when the voltages stagnate.
        """
        self.dt = dt
        self.num_iterations = num_iterations
        self.resistance = resistance
        self.capacitance = capacitance
        self.stop_criterion = stop_criterion
        self.initial_temp_cont = initial_temp_cont
        self.end_temp_cont = end_temp_cont
        self.coupling = coupling

    def mosfet(self, voltage: np.ndarray):
        pass

    def dvdt(
        self,
        t: float,
        vt: np.ndarray,
    ):
        """Differential equations for the multiplicative BRIM model when flipping is involved.

        Args:
            t (float): time
            vt (np.ndarray): current voltages
            coupling (np.ndarray): coupling matrix J

        Returns:
            dv (np.ndarray): the change of the voltages
        """
        t = np.float32(t)
        vt = vt.astype(np.float32)
        coupling = self.coupling.astype(np.float32)
        return dvdt_solver(t, vt, coupling, np.int8(self.bias), np.float32(self.capacitance))

    def noise(self, Temp: float, voltages: np.ndarray):
        if Temp != 0.0:
            noise = Temp * (np.random.normal(scale=1 / 1.96, size=voltages.shape))
            cond1 = (voltages >= 1) & (noise > 0)
            cond2 = (voltages <= -1) & (noise < 0)
            noise *= np.where(cond1 | cond2, 0.0, 1.0)
        else:
            noise = np.zeros_like(voltages)
        return noise

    def inner_loop(self,
                   model:IsingModel,
                   state: np.ndarray,
                   log: HDF5Logger):
        # Set up the simulation
        i = 0
        tk = 0
        max_change = np.inf
        Temp = self.initial_temp_cont if self.initial_temp_cont < 1.0 else 0.5
        cooling_rate = (
            return_rx(self.num_iterations, self.initial_temp_cont, self.end_temp_cont)
            if self.initial_temp_cont != 0.0
            else 1.0
        )
        previous_voltages = state.copy()
        energy = model.evaluate(np.sign(state[: model.num_variables]))
        log.log(time_clock=0.0, energy=energy, state=np.sign(state), voltages=state)
        count = np.zeros((model.num_variables,))
        norm_prev = np.linalg.norm(previous_voltages, ord=np.inf)
        while i < self.num_iterations and max_change > self.stop_criterion:
            k1 = self.dt * self.dvdt(tk, previous_voltages)
            k2 = self.dt * self.dvdt(tk + self.dt, previous_voltages + k1)
            k3 = self.dt * self.dvdt(tk + 0.5 * self.dt, previous_voltages + 0.25 * (k1 + k2))

            noise = self.noise(Temp, previous_voltages)
            new_voltages = previous_voltages + (k1 + k2 + 4.0 * k3) / 6.0 + noise
            Temp *= cooling_rate
            tk += self.dt
            i += 1

            # Log everything
            sample = np.sign(new_voltages[: model.num_variables])
            energy = model.evaluate(sample)
            count += np.where(
                (previous_voltages[: model.num_variables] * new_voltages[: model.num_variables] < 0), 1, 0
            )
            # if i % 10000 == 0:
            #     LOGGER.info(energy)
            log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[: model.num_variables])

            # Only compute norm if needed
            if i > 0:
                diff = np.abs(new_voltages - previous_voltages)
                max_change = np.max(diff) / (norm_prev if norm_prev != 0 else 1)
                norm_prev = np.linalg.norm(new_voltages, ord=np.inf)
            previous_voltages = new_voltages.copy()
        return np.sign(new_voltages[:model.num_variables]), energy

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
        # c = np.where(np.abs(vt) > 1., 1 / np.abs(vt), vt)

        # ZIV diode
        z = vt / self.resistance * (vt - 1) * (vt + 1) * self.mu_param

        # Compute the voltage change dv
        dv = 1 / self.capacitance * (np.dot(coupling,  np.sign(vt)) - z)

        # Ensure the voltages stay in the range [-1, 1]
        cond1 = (dv > 0) & (vt > 0)
        cond2 = (dv < 0) & (vt < 0)
        dv *= np.where(cond1 | cond2, (1-vt**2), 1.0)

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
        nb_flipping: int,
        cluster_threshold:float,
        init_cluster_size:float,
        end_cluster_size:float,
        resistance: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        initial_temp_cont: float = 1.0,
        end_temp_cont: float = 0.05,
        stop_criterion: float = 1e-8,
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
            file (pathlib.Path, None, optional): the path to the logfile. Defaults to None.

        Returns:
            tuple[np.ndarray, float]: the best energy and the best sample.
        """

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            self.bias = True
        else:
            new_model = model
            self.bias = False

        # Ensure the mean and variance of J are reasonable
        alpha = 1.0
        coupling = alpha * triu_to_symm(new_model.J) * 1 / resistance

        # Set the parameters for easy calling
        init_size = int(init_cluster_size * model.num_variables)
        end_size = int(end_cluster_size * model.num_variables)
        self.set_params(
            dtMult,
            num_iterations,
            resistance,
            capacitance,
            stop_criterion,
            initial_temp_cont,
            end_temp_cont,
            coupling,
        )

        # make sure the correct random seed is used
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Set up the bias node and add noise to the initial voltages
        num_var = model.num_variables
        if self.bias:
            v = np.empty(num_var + 1, dtype=np.float32)
            v[:-1] = initial_state
            v[-1] = 1.0
        else:
            v = initial_state.astype(np.float32, copy=True)

        # Schema for logging
        schema = {
            "time_clock": float,
            "energy": np.float32,
            "state": (np.int8, (num_var,)),
            "voltages": (np.float32, (num_var,)),
        }

        with HDF5Logger(file, schema) as log:
            self.log_metadata(
                logger=log,
                initial_state=np.sign(v[:-1]),
                model=model,
                num_iterations=num_iterations,
                time_step=dtMult,
                temperature=initial_temp_cont,
            )
            best_energy = np.inf
            best_sample = v[:model.num_variables].copy()
            size_func = self.cluster_size(init_size, end_size, nb_flipping)
            for it in range(nb_flipping):

                sample, energy, count = self.inner_loop(model, v, log)

                if energy < best_energy:
                    best_energy = energy
                    best_sample = sample.copy()

                cluster = self.find_cluster(count, size_func(it), cluster_threshold)
                v = best_sample.copy()
                v[cluster] *= -1
                if self.bias:
                    v = np.block([v, 1.0])

            LOGGER.info(f"Finished with energy: {energy}")
            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=dtMult * num_iterations)
        return best_sample, best_energy

    def cluster_size(self, init_size:int, end_size:int, num_iterations:int) -> callable:
        return lambda x: int((return_rx(num_iterations, init_size, end_size)**(x*3)) * (init_size-end_size) + end_size)

    def find_cluster(self, counts: np.ndarray, cluster_size: int, cluster_threshold: float):
        """Finds the cluster of nodes to flip. These nodes are chosen based on the frequency of flipping.

        Args:
            counts (np.ndarray): the amount of times each node changed their sign during convergence.
            cluster_size (int): the size of the cluster to find.
            cluster_threshold (float): the threshold for selecting nodes.
        """
        freq = counts / (np.max(np.abs(counts)) if np.max(np.abs(counts)) != 0 else 1)

        available_nodes = np.where(freq < cluster_threshold)[0]
        current_size = len(available_nodes)
        if len(available_nodes) <= cluster_size:
            ind_unavailable_nodes = np.where(freq >= cluster_threshold)[0]
            chosen_nodes = np.array([], dtype=int)
            while len(chosen_nodes) < cluster_size - current_size:
                chosen_nodes = np.unique(
                    np.append(
                        chosen_nodes,
                        np.random.choice(ind_unavailable_nodes, (cluster_size - current_size - len(chosen_nodes),)),
                    )
                )
            available_nodes = np.append(available_nodes, chosen_nodes)
            cluster = available_nodes
        else:
            cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
        return cluster
