import numpy as np
import pathlib
import time

# from ising.stages import LOGGER
from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.numpy import triu_to_symm


class Multiplicative(SolverBase):
    def __init__(self, adjustments=False):
        super().__init__()
        self.name = "Multiplicative"
        self.adjustments = adjustments

    def set_params(
        self,
        num_variables: int,
        dt: float,
        num_iterations: int,
        capacitance: float,
        current: float,
        stop_criterion: float,
        coupling: np.ndarray,
        # coupling_pos: np.ndarray,
        # coupling_neg: np.ndarray,
        voltage_delay_idx: np.ndarray | None = None,
    ):
        """Set the parameters for the solver.

        @type num_variables: int
        @param model: model to solve
        @type dt: float
        @param dt: time step.
        @type num_iterations: int
        @param num_iterations: the number of iterations.
        @type capacitance: float
        @param capacitance: the capacitance of the system.
        @type current: float
        @param current: the unit current that flows through the cells.
        @type stop_criterion: float
        @param stop_criterion: the stopping criterion to stop the solver when the voltages stagnate.
        @type coupling: np.ndarray
        @param coupling: the coupling matrix of the system.
        @type coupling_pos: np.ndarray
        @param coupling_pos: the coupling matrix of the system for positive pushes.
        @type coupling_neg: np.ndarray
        @param coupling_neg: the coupling matrix of the system for negative pushes.
        @type voltage_delay_idx: np.ndarray|None
        @param voltage_delay_idx: the matrix containing the delay indices for each voltage.
        """
        self.num_variables = num_variables
        self.dt = np.float32(dt)
        self.num_iterations = num_iterations
        self.capacitance = np.float32(capacitance)
        self.current = current
        self.stop_criterion = stop_criterion
        self.coupling = coupling.astype(np.float32)
        # self.coupling_pos = coupling_pos.astype(np.float32)
        # self.coupling_neg = coupling_neg.astype(np.float32)
        self.voltage_delay_idx = voltage_delay_idx
        self.tau_system = capacitance / (current * np.mean(np.sum(coupling, axis=1)))

    def construct_voltage_delay(self, previous_states: np.ndarray) -> np.ndarray:
        """Generates a matrix of voltages taking into account what each voltage sees at the current time step.

        @type previous_states: np.ndarray
        @param previous_states: deque containing all the previous states up to\
                  accumulation_delay + broadcast_delay + 1 time steps.
        @type previous_Voltages: np.ndarray|None
        @param previous_Voltages: list containing all previous voltage matrices up to inter_delay\
             time steps.
        @rtype: np.ndarray
        @return: voltage matrix with all delays taken into account.
        """
        Voltages = np.zeros(
            (self.num_variables + int(self.bias), self.num_variables + int(self.bias)), dtype=np.float32
        )

        Voltages[: self.num_variables, : self.num_variables] = previous_states[
            self.voltage_delay_idx, np.arange(self.num_variables)[:, None]
        ]
        if self.bias == 1:
            Voltages[:, -1] = previous_states[0, :]
            Voltages[-1, :] = 1.0
        return Voltages

    def inner_loop_FE(
        self,
        model: IsingModel,
        state: np.ndarray,
        total_delay: int,
        logging: HDF5Logger | None = None,
    ) -> tuple[np.ndarray, float]:
        """Simulates the hardware with the Forward Euler method.

        @type model: IsingModel
        @param model: the model to solve.
        @type state: np.ndarray
        @param state: the initial state to start the simulation.
        @type total_delay: int
        @param total_delay: total delay present in the system.
        @type logging: HDF5Logger|None
        @param logging: the logger to use when flipping is disabled.
        @rtype: tuple[np.ndarray, float]
        @return: the new state and new energy.
        """
        # set up the simulation
        i = 0
        max_change = np.inf
        norm_prev = np.linalg.norm(state, ord=np.inf)

        if logging is not None:
            logging.log(
                time=0.0,
                state=np.sign(state[: model.num_variables]),
                energy=model.evaluate(np.sign(state[: model.num_variables]).astype(np.float32)),
                voltages=state[: model.num_variables],
            )

        # Set up new voltages
        new_state = state.copy().astype(np.float32)

        # States needed for delay calculation. The newest state is always appended to the end of the list.
        previous_states = np.array([np.sign(state) for _ in range(total_delay + 1)])
        counter = total_delay + 1
        dv = self.coupling @ np.sign(state) * self.current / self.capacitance
        time_zero = 0.0
        nb_operations = 0
        while i < self.num_iterations and max_change > self.stop_criterion:
            if counter < total_delay + 1:
                if total_delay > 0:
                    Voltages = self.construct_voltage_delay(previous_states)
                    dv = np.diagonal(self.coupling @ Voltages).copy()
                else:  # no hardware imperfections
                    dv = self.coupling @ np.sign(state)
                dv *= self.current / self.capacitance
                counter += 1
                if self.bias:
                    dv[-1] = 0.0
            dv[self.freeze_nodes] = 0.0
            new_state = np.clip(state + self.dt * dv, -1, 1)

            if np.max(np.sign(new_state) != np.sign(state)):
                counter = 0

            if i > 0 and (i % 10) == 0:
                diff = np.abs(new_state - previous_states[-1])
                norm_prev = np.linalg.norm(previous_states[-1])
                max_change = np.max(diff) / (norm_prev if norm_prev != 0 else 1)

            previous_states = np.block([[np.sign(new_state)], [previous_states]])[:-1, :]
            state = new_state.copy()
            i += 1
            if logging is not None:
                logging.log(
                    time=i * self.dt,
                    state=np.sign(new_state[: model.num_variables]),
                    energy=model.evaluate(np.sign(new_state[: model.num_variables]).astype(np.float32)),
                    voltages=new_state[: model.num_variables],
                )
            if i * self.dt - time_zero >= self.tau_system:
                time_zero = i * self.dt
                nb_operations += 2 * model.num_variables**2 + 3 * model.num_variables
        return (
            np.where(new_state[: model.num_variables] >= 0, 1, -1).astype(np.float32),
            model.evaluate(np.where(new_state[: model.num_variables] >= 0, 1, -1).astype(np.float32)),
            i * self.dt,
            nb_operations,
        )

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        num_iterations: int,
        nb_flipping: int,
        cluster_threshold: float,
        init_cluster_size: float,
        end_cluster_size: float,
        exponent: float = 3.0,
        cluster_choice: str = "random",
        current: float = 1.0,
        capacitance: float = 1.0,
        seed: int = 0,
        stop_criterion: float = 1e-8,
        accumulation_delay: float = 0.0,
        broadcast_delay: float = 0.0,
        delay_offset: float = 0.0,
        combine_nodes: bool = False,
        nb_splits: int = 2,
        # sigma_J: float = -1.0,
        file: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float, float, int, int]:
        """Solves the given problem using a multiplicative coupling scheme.

        @type model: IsingModel
        @param model: the model to solve.
        @type initial_state: np.ndarray
        @param initial_state: the initial spins of the nodes.
        @type num_iterations: int
        @param num_iterations: the number of iterations.
        @type freeze_spins: list[int]
        @param freeze_spins: indices of spins not allowed to change. These also include the replicated spins when\
            combine_nodes is True.
        @type nb_flipping: int
        @param nb_flipping: the number of flipping iterations.
        @type cluster_threshold: float
        @param cluster_threshold: the threshold for clustering.
        @type init_cluster_size: float
        @param init_cluster_size: the initial cluster size.
        @type end_cluster_size: float
        @param end_cluster_size: the final cluster size.
        @type exponent: float
        @param exponent: the exponent for the exponential decrease of the cluster size.
        @type cluster_choice: str
        @param cluster_choice: the choice of clustering method.
        @type current: float
        @param current: the current flowing through a coupling unit.
        @type capacitance: float
        @param capacitance: the capacitance of the system.
        @type seed: int
        @param seed: the seed for random number generation.
        @type stop_criterion: float
        @param stop_criterion: the stopping criterion to stop the solver when the voltages don't change
        @type accumulation_delay: float
        @param accumulation_delay: the amount of accumulation delay in percentage of C/I value.
        @type  broadcast_delay: float
        @param broadcast_delay: the amount of broadcast delay in percentage of C/I value.
        @type delay_offset: float
        @param delay_offset: amount of delay due to the comparator, which offsets all the delays.
        @type sigma_J: float
        @param sigma_J: the standard deviation of mismatch in the coupling.
        @type combine_nodes: bool
        @param combine_nodes: whether nodes need to be flipped together.
        @type nb_splits: int
        @param nb_splits: if combine_nodes is True, the number of nodes that need to be flipped together.
        @type file: pathlib.Path|None
        @param file: the path to the logfile
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: the final state, final energy, total computation time, number of operations, and number of iterations\
              until convergence.
        """
        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model:IsingModel = model.transform_to_no_h()
            self.bias = np.int8(1)
        else:
            new_model:IsingModel = model
            self.bias = np.int8(0)
        num_variables = model.num_variables
        self.freeze_nodes = model.freeze_spins
        coupling = triu_to_symm(new_model.J)
        # Include J mismatch
        # if sigma_J != -1.0:
        #     self.mismatch = True
        #     coupling_pos = coupling * (1 + np.random.normal(0.0, sigma_J, coupling.shape))
        #     coupling_neg = coupling * (1 + np.random.normal(0.0, sigma_J, coupling.shape))
        # else:
        #     self.mismatch = False
        #     coupling_pos = coupling
        #     coupling_neg = coupling

        # Change time step according to parameters of system
        dtMult = 0.1 * capacitance / (current * np.max(np.abs(np.sum(coupling, axis=1))))

        # Set up delay
        capacitance_delay = capacitance / num_variables
        time_constant = capacitance_delay / current

        if accumulation_delay > 0.0 and accumulation_delay * time_constant < dtMult:
            dtMult = accumulation_delay * time_constant
            num_iterations = int(np.ceil(num_iterations * (dtMult / (accumulation_delay * time_constant))))
        if broadcast_delay > 0.0 and broadcast_delay * time_constant < dtMult:
            dtMult = broadcast_delay * time_constant
            num_iterations = int(np.ceil(num_iterations * (dtMult / (broadcast_delay * time_constant))))
        if delay_offset > 0.0 and delay_offset * time_constant < dtMult:
            dtMult = delay_offset * time_constant
            num_iterations = int(np.ceil(num_iterations * (dtMult / (delay_offset * time_constant))))

        accumulation_delay = int(accumulation_delay * time_constant / dtMult)
        broadcast_delay = int(broadcast_delay * time_constant / dtMult)
        delay_offset = int(delay_offset * time_constant / dtMult)

        total_delay = (num_variables - 1) * (accumulation_delay + broadcast_delay) + delay_offset

        if accumulation_delay > 0 or broadcast_delay > 0 or delay_offset > 0:
            voltage_delay_idx = np.zeros((num_variables, num_variables), dtype=np.int8)
            for i in range(num_variables):
                for j in range(num_variables):
                    voltage_delay_idx[i, j] = (
                        np.floor(np.abs(i - j) * (accumulation_delay + broadcast_delay)) + delay_offset
                    )

        # Set the parameters for easy calling
        if combine_nodes:
            num_var = int((num_variables - len(self.freeze_nodes)) / nb_splits)
            init_size = int(init_cluster_size * num_var)
            end_size = int(end_cluster_size * num_var)
        else:
            init_size = int(init_cluster_size * model.num_non_frozen_variables)
            end_size = int(end_cluster_size * model.num_non_frozen_variables)
        if end_size < 1:
            end_size = 1
        self.set_params(
            num_variables,
            dtMult,
            num_iterations,
            capacitance,
            current,
            stop_criterion,
            coupling,
            # coupling_pos,
            # coupling_neg,
            voltage_delay_idx=voltage_delay_idx if (total_delay > 0) else None,
        )

        # make sure the correct random seed is used
        np.random.seed(seed)
        self.generator = np.random.choice

        # Set up the bias node and add noise to the initial voltages
        if self.bias:
            v = np.empty(num_variables + 1, dtype=np.float32)
            v[:-1] = initial_state
            v[-1] = 1.0
        else:
            v = initial_state.astype(np.float32, copy=True)
        # Schema for logging
        if nb_flipping == 1:
            schema = {
                "time": np.float32,
                "energy": np.float32,
                "state": (np.int8, (num_variables,)),
                "voltages": (np.float32, (num_variables,)),
            }
        else:
            schema = {
                "energy_best": np.float32,
                "energy": np.float32,
                "state_out": (np.int8, (num_variables,)),
                "state_in": (np.int8, (num_variables,)),
                "cluster": (np.int8, (num_variables,)),
            }

        # Define cluster function
        if cluster_choice == "random":
            find_cluster = self.find_cluster_random
        else:
            raise ValueError(
                f" Unknown cluster choice: {cluster_choice}. \
             Currently supported: random, gradient, weighted_mean_smallest, weighted_mean_largest."
            )
        additional_information = {
            "current_state": np.ndarray,
            "cluster_threshold": cluster_threshold,
            "optimal_points": [],
            "choice": cluster_choice,
        }

        with HDF5Logger(file, schema) as log:
            if log.filename is not None:
                self.log_metadata(
                    logger=log,
                    initial_state=np.sign(initial_state),
                    model=model,
                    num_iterations=num_iterations,
                    time_step=dtMult,
                    cluster_choice=cluster_choice,
                    exponent=exponent,
                )
                if nb_flipping > 1:
                    log.log(
                        energy_best=np.inf,
                        energy=np.inf,
                        state_in=np.sign(v[: num_variables]),
                        state_out=np.zeros(num_variables, dtype=np.int8),
                        cluster=np.zeros(num_variables, dtype=np.int8),
                    )
            best_energy = np.inf
            best_sample = v[: num_variables].copy()
            if nb_flipping == 1:
                logging = log
            else:
                logging = None

            counter = 0  # Counter for no improvement
            restart = 0  # When counter reaches threshold, size of cluster is reset with restart = it
            tot_time = 0  # total time of system (analog + flipping)
            tot_ops = 0  # total amount of operations (analog + flipping)
            for it in range(nb_flipping):
                sample, energy, ana_time, ana_ops = self.inner_loop_FE(model, v, total_delay, logging)
                start = time.time()
                additional_information["current_state"] = sample[
                    np.setdiff1d(np.arange(num_variables), self.freeze_nodes)
                ]

                if energy < best_energy:
                    best_energy = energy
                    best_sample = sample.copy()
                    additional_information["optimal_points"].append((best_sample.copy(), best_energy))
                    counter = 0
                else:
                    counter += 1
                # if counter >= int(nb_flipping / 4):
                #     restart = int(it / 2)
                #     counter = 0
                cluster, operations = find_cluster(
                    self.size_function(
                        iteration=it - restart,
                        total_iterations=nb_flipping + int(nb_flipping == 1),
                        init_size=init_size,
                        end_size=end_size,
                        exponent=exponent,
                    ),
                    combine_nodes=combine_nodes,
                    nb_splits=nb_splits,
                    **additional_information,
                )
                v = best_sample.copy()
                v[cluster] *= np.float32(-1.0)
                if self.bias:
                    v = np.block([v, np.float32(1.0)])
                # Log everything
                if log.filename is not None and nb_flipping > 1:
                    log.log(
                        energy_best=best_energy,
                        energy=energy,
                        state_out=sample,
                        state_in=best_sample,
                        cluster=np.where(v[: num_variables] == best_sample, 0, 1).astype(np.int8),
                    )
                tot_time += ana_time + time.time() - start
                tot_ops += (
                    ana_ops +                                                   # analog operation count
                    2 * num_variables**2 + 3 * num_variables +      # operation count for energy calculation
                    operations +                                                # cluster choice operation count
                    len(cluster)                                                # set cluster operation count
                )
            if log.filename is not None:
                log.write_metadata(
                    solution_state=sample,
                    solution_energy=energy,
                    total_time=dtMult * num_iterations,
                )
        return best_sample, best_energy, tot_time, tot_ops, nb_flipping

    def size_function(
        self,
        iteration: int,
        total_iterations: int,
        init_size: int,
        end_size: int,
        exponent: float = 3.0,
    ):
        result = np.floor(
            (((end_size - 1) / init_size) ** (iteration * exponent / (total_iterations - 1))) * (init_size - end_size)
            + end_size
        )

        return int(result)

    # def find_cluster_gradient(
    #     self, cluster_size: int, combine_nodes: bool, nb_splits: int, **additional_information
    # ) -> np.ndarray:
    #     coupling = self.coupling_d * self.resistance
    #     sigma = additional_information["current_state"]

    #     gradient = (coupling @ np.block([sigma, 1]))[: len(sigma)]
    #     gradient /= np.max(gradient)
    #     if combine_nodes:
    #         num_nodes = (len(sigma)) / nb_splits
    #        gradient = np.array([np.sum(gradient[i * nb_splits : (i + 1) * nb_splits]) for i in range(int(num_nodes))])
    #     else:
    #         num_nodes = len(sigma)
    #     threshold = additional_information["cluster_threshold"]

    #     available_nodes = np.where(
    #         gradient >= threshold, np.setdiff1d(np.arange(num_nodes), self.freeze_nodes), -1
    #     )  # Chosen nodes based on threshold
    #     if len(available_nodes[available_nodes >= 0]) < cluster_size:  # Case when not enough nodes are available
    #         current_size = len(available_nodes[available_nodes >= 0])
    #         ind_unavailable_nodes = np.where(available_nodes < 0)[0]
    #         chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
    #         available_nodes[chosen_nodes] = np.arange(num_nodes)[chosen_nodes]
    #         cluster = available_nodes[available_nodes >= 0]
    #     else:  # case when enough nodes are available
    #         cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(cluster_size,), replace=False)
    #     if combine_nodes:
    #         cluster = np.array([nb_splits * cluster_elem + i for cluster_elem in cluster for i in range(nb_splits)])
    #     return cluster

    def find_cluster_random(
        self, cluster_size: int, combine_nodes: bool, nb_splits: int, **additional_information
    ) -> tuple[np.ndarray, int]:
        """Finds a random cluster of nodes to flip.

        @type cluster_size: int
        @param cluster_size: the size of the cluster to find.
        @rtype: tuple[np.ndarray, int]
        @return: the indices of the nodes in the cluster and the amount of operations.
        """
        if combine_nodes:
            cluster = self.generator(
                np.setdiff1d(np.arange(int(self.num_variables / nb_splits)), self.freeze_nodes[::nb_splits]),
                size=(cluster_size,),
                replace=False,
            )
            state = additional_information["current_state"]
            cluster_nodes = []
            for cluster_elem in cluster:
                replica_indices = nb_splits * cluster_elem + np.arange(nb_splits)
                replica_states = state[replica_indices]
                if np.all(replica_states == replica_states[0]) or not self.adjustments:
                    # Replicas agree: flip the whole group.
                    cluster_nodes.extend(replica_indices)
                else:
                    # Replicas disagree: flip the minority so the group ends up agreeing.
                    target = np.sign(np.sum(replica_states))
                    if target == 0:  # perfect tie: fall back to a reference replica
                        target = replica_states[0]
                    cluster_nodes.extend(replica_indices[replica_states != target])
            cluster = np.array(cluster_nodes, dtype=int)
        else:
            cluster = self.generator(
                np.setdiff1d(np.arange(int(self.num_variables)), self.freeze_nodes),
                size=(cluster_size,),
                replace=False,
            )
        return cluster, self.num_variables

    # def find_cluster_weighted_mean(
    #     self, cluster_size: int, combine_nodes: bool, nb_splits: int, **additional_information
    # ) -> np.ndarray:
    #     optimal_points = additional_information["optimal_points"]
    #     choice = additional_information["choice"]
    #     weight_nodes = np.zeros_like(optimal_points[0][0], dtype=float)

    #     for point, en in optimal_points:
    #         weight_nodes += 1 / en * point  # the smaller the energy, the larger the weight
    #     if np.linalg.norm(weight_nodes) == 0:
    #         weight_nodes = np.random.random(weight_nodes.shape)  # First step is random choice
    #     if combine_nodes:
    #         weight_nodes = np.array(
    #             [
    #                 np.sum(weight_nodes[i * nb_splits : (i + 1) * nb_splits])
    #                 for i in range(int(len(weight_nodes) / nb_splits))
    #             ]
    #         )
    #     weight_nodes = np.abs(weight_nodes) / np.max(np.abs(weight_nodes))
    #     if choice == "smallest":
    #         available_nodes = np.where(weight_nodes < additional_information["cluster_threshold"])[0]
    #         current_size = len(available_nodes)
    #         if len(available_nodes) < cluster_size:
    #             ind_unavailable_nodes = np.where(weight_nodes >= additional_information["cluster_threshold"])[0]
    #             chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
    #             available_nodes = np.append(available_nodes, chosen_nodes)
    #     else:
    #         available_nodes = np.where(weight_nodes > additional_information["cluster_threshold"])[0]
    #         current_size = len(available_nodes)
    #         if len(available_nodes) < cluster_size:
    #             ind_unavailable_nodes = np.where(weight_nodes <= additional_information["cluster_threshold"])[0]
    #             chosen_nodes = np.random.choice(ind_unavailable_nodes, (cluster_size - current_size,), replace=False)
    #             available_nodes = np.append(available_nodes, chosen_nodes)
    #     cluster = np.random.choice(available_nodes, size=(cluster_size,), replace=False)
    #     if combine_nodes:
    #         cluster = np.array([nb_splits * cluster_elem + i for cluster_elem in cluster for i in range(nb_splits)])
    #     return cluster
