import numpy as np
import pathlib
from pylfsr import LFSR

# from ising.stages import LOGGER
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
        self.dt = np.float32(dt)
        self.num_iterations = num_iterations
        self.resistance = np.float32(resistance)
        self.capacitance = np.float32(capacitance)
        self.stop_criterion = stop_criterion
        self.initial_temp_cont = initial_temp_cont
        self.end_temp_cont = end_temp_cont
        self.coupling_d = coupling.astype(np.float32)
        self.quarter = np.float32(0.25)
        self.half = np.float32(0.5)
        self.four = np.float32(4.0)
        self.six = np.float32(6.0)

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
        return dvdt_solver(t, vt, self.coupling_d, np.int8(self.bias), self.capacitance)

    def inner_loop(self, model: IsingModel, state: np.ndarray):
        """! Simulates the hardware

        @param model (IsingModel): the model to solve.
        @param state (np.ndarray): the initial state to start the simulation.

        @return sigma (np.ndarray): the final discrete state of the system.
        @return energy (float): the final energy of the system.
        @return count (np.ndarray): the count of sign flips for every node.
        """

        # Set up the simulation
        i = 0
        tk = np.float32(0.0)
        max_change = np.inf

        previous_voltages = state.astype(np.float32)

        count = np.zeros((model.num_variables,))
        norm_prev = np.linalg.norm(previous_voltages, ord=np.inf)
        while i < self.num_iterations and max_change > self.stop_criterion:
            # LOGGER.info(f"Iteration {i}")
            k1 = self.dt * self.dvdt(tk, previous_voltages)
            k2 = self.dt * self.dvdt(tk + self.dt, previous_voltages + k1)
            k3 = self.dt * self.dvdt(tk + self.half * self.dt, previous_voltages + self.quarter * (k1 + k2))

            new_voltages = previous_voltages + (k1 + k2 + self.four * k3) / self.six

            tk += self.dt
            i += 1

            count += (previous_voltages[: model.num_variables] * new_voltages[: model.num_variables]) < 0


            # Only compute norm if needed
            if i > 0 and i % 1000:
                diff = np.abs(new_voltages - previous_voltages)
                max_change = np.max(diff) / (norm_prev if norm_prev != 0 else 1)
                norm_prev = np.linalg.norm(new_voltages, ord=np.inf)
            previous_voltages = new_voltages.copy()

        energy = model.evaluate(np.sign(new_voltages[: model.num_variables], dtype=np.float32))

        return np.sign(new_voltages[: model.num_variables]), energy, count

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        dtMult: float,
        num_iterations: int,
        nb_flipping: int,
        cluster_threshold: float,
        init_cluster_size: float,
        end_cluster_size: float,
        cluster_choice: str = "random",
        pseudo_length: int | None = None,
        resistance: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        initial_temp_cont: float = 0.0,
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
            self.bias = np.int8(1)
        else:
            new_model = model
            self.bias = np.int8(0)

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
        if pseudo_length is not None:
            degree = np.log2(pseudo_length + 1)
            if degree not in range(1, 13):
                raise ValueError("pseudo_length should be of the form 2^n-1 with n an integer between 1 and 12.")
            fpoly = [degree]
            if degree == 5:
                fpoly.append(2)
            elif degree == 8:
                fpoly += [7, 2, 1]
            elif degree == 9:
                fpoly.append(4)
            elif degree == 10:
                fpoly.append(3)
            elif degree == 11:
                fpoly.append(2)
            elif degree == 12:
                fpoly += [6, 4, 1]
            else:
                fpoly.append(1)
            self.generator = LFSR(fpoly=fpoly, initstate="random").runKCycle
            nb_bits = int(np.log2(model.num_variables)+1)
        else:
            np.random.seed(seed)
            self.generator = np.random.choice
            nb_bits = -1
            pseudo_length = -1

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
            "energy": np.float32,
            "state": (np.int8, (num_var,)),
        }

        # Define cluster function
        choice = ""
        if cluster_choice == "random":
            find_cluster = self.find_cluster_random
        elif cluster_choice == "gradient":
            find_cluster = self.find_cluster_gradient
        elif cluster_choice == "weighted_mean_smallest":
            find_cluster = self.find_cluster_weighted_mean
            choice = "smallest"
        elif cluster_choice == "weighted_mean_largest":
            find_cluster = self.find_cluster_weighted_mean
            choice = "largest"
        elif cluster_choice == "frequency":
            find_cluster = self.find_cluster_frequency
        else:
            raise ValueError(
                f" Unknown cluster choice: {cluster_choice}. \
             Currently supported: random, gradient, weighted_mean_smallest, weighted_mean_largest, frequency."
            )
        additional_information = {
            "count": np.ndarray,
            "current_state": np.ndarray,
            "cluster_threshold": cluster_threshold,
            "optimal_points": [],
            "choice": choice,
            "pseudo_length": pseudo_length,
            "nb_bits": nb_bits,
        }

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(initial_state),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtMult,
                    temperature=initial_temp_cont,
                    pseudo_length=pseudo_length,
                    cluster_choice=cluster_choice
                )
            best_energy = np.inf
            best_sample = v[: model.num_variables].copy()
            for it in range(nb_flipping):
                sample, energy, count = self.inner_loop(model, v)
                additional_information["count"] = count
                additional_information["current_state"] = sample
                if energy < best_energy:
                    best_energy = energy
                    best_sample = sample.copy()
                    additional_information["optimal_points"].append((best_sample.copy(), best_energy))

                cluster = find_cluster(
                    self.size_function(
                        iteration=it,
                        total_iterations=num_iterations,
                        init_size=init_size,
                        end_size=end_size,
                    ),
                    **additional_information,
                )
                v = best_sample.copy()
                v[cluster] *= -1
                if self.bias:
                    v = np.block([v, np.float32(1.0)])

                # Log everything
                if log.filename is not None:
                    log.log(energy=best_energy, state=best_sample)

            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_time=dtMult * num_iterations,
                )
        return best_sample, best_energy

    def size_function(
        self,
        iteration: int,
        total_iterations: int,
        init_size: int,
        end_size: int,
        exponent: float = 3.0,
    ):
        return int(
            (return_rx(total_iterations, init_size, end_size) ** (iteration * exponent)) * (init_size - end_size)
            + end_size
        )

    def find_cluster_gradient(self, cluster_size: int, **additional_information)->np.ndarray:
        coupling = self.coupling_d * self.resistance
        sigma = additional_information["current_state"]
        threshold = additional_information["cluster_threshold"]

        gradient = coupling @ np.block([sigma, 1])
        gradient /= np.max(gradient)
        available_nodes = np.where(gradient >= threshold, np.arange(len(sigma)), -1)  # Chosen nodes based on threshold
        if len(available_nodes[available_nodes >= 0]) < cluster_size:  # Case when not enough nodes are available
            current_size = len(available_nodes[available_nodes >= 0])
            ind_unavailable_nodes = np.where(available_nodes < 0)[0]
            chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
            available_nodes[chosen_nodes] = np.arange(len(sigma))[chosen_nodes]
            cluster = available_nodes[available_nodes >= 0]
        else:  # case when enough nodes are available
            cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(cluster_size,), replace=False)
        return cluster

    def find_cluster_random(self, cluster_size: int, **additional_information) -> np.ndarray:
        """Finds a random cluster of nodes to flip.

        Args:
            cluster_size (int): the size of the cluster to find.
        """
        if additional_information["pseudo_length"] == -1:
            cluster = self.generator(np.arange(0, self.coupling_d.shape[0]), size=(cluster_size,), replace=False)
        else:
            cluster = set()
            while len(cluster) < cluster_size:
                nb_bits = (cluster_size - len(cluster)) * additional_information["nb_bits"]
                seq = np.array(self.generator(nb_bits)).reshape(
                    (cluster_size - len(cluster), additional_information["nb_bits"])
                )
                for row in seq:
                    str_bin = np.array2string(row, separator="")[1:-1]
                    node = int(str_bin, 2)
                    if node < self.coupling_d.shape[0]:
                        cluster.add(node)
            cluster = np.array(list(cluster))

        return cluster

    def find_cluster_weighted_mean(self, cluster_size: int, **additional_information)-> np.ndarray:
        optimal_points = additional_information["optimal_points"]
        choice = additional_information["choice"]
        weight_nodes = np.zeros_like(optimal_points[0], dtype=float)
        for point, en in optimal_points:
            weight_nodes += 1 / en * point  # the smaller the energy, the larger the weight
        if np.linalg.norm(weight_nodes) == 0:
            weight_nodes = np.random.random(weight_nodes.shape)  # First step is random choice
        if choice == "smallest":
            cluster = np.argsort(np.abs(weight_nodes))[:cluster_size]
        else:
            cluster = np.argsort(np.abs(weight_nodes))[-cluster_size:]

        return cluster

    def find_cluster_frequency(self, cluster_size: int, **additional_information)->np.ndarray:
        """Finds the cluster of nodes to flip. These nodes are chosen based on the frequency of flipping.

        Args:
            counts (np.ndarray): the amount of times each node changed their sign during convergence.
            cluster_size (int): the size of the cluster to find.
            cluster_threshold (float): the threshold for selecting nodes.
        """
        counts = additional_information["counts"]
        cluster_threshold = additional_information["cluster_threshold"]

        freq = counts / (np.max(np.abs(counts)) if np.max(np.abs(counts)) != 0 else 1)

        available_nodes = np.where(freq < cluster_threshold)[0]
        current_size = len(available_nodes)
        if len(available_nodes) < cluster_size:
            ind_unavailable_nodes = np.where(freq >= cluster_threshold)[0]
            chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
            available_nodes = np.append(available_nodes, chosen_nodes)
            cluster = available_nodes
        else:
            cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
        return cluster
