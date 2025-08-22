from ising.stages import LOGGER
from typing import Any
import networkx as nx
import numpy as np
from argparse import Namespace

from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel
from ising.generators.TSP import TSP
from ising.stages.qkp_parser_stage import QKPParserStage

class DummyCreatorStage(Stage):
    """! Stage to create a dummy Ising model for testing purposes.
    To create a dummy model, the problem type and size must be specified in the yaml configuration.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, config: Any, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.problem_type = config.problem_type

    def run(self) -> Any:
        """! Creates a dummy Ising model."""

        LOGGER.info(f"Creating a dummy {self.problem_type} model.")
        dummy_creator = self.config.dummy_creator if hasattr(self.config, "dummy_creator") else False
        if dummy_creator:
            seed = self.config.dummy_seed

            if self.problem_type == "Maxcut":
                N = self.config.dummy_size
                dummy_dict = self.generate_dummy_maxcut(N, seed)
            elif self.problem_type in ["TSP", "ATSP"]:
                N = self.config.dummy_size
                weight_constant = self.config.dummy_weight_constant if hasattr(
                    self.config, "dummy_weight_constant") else 0.0
                if not hasattr(self.config, "weight_constant"):
                    LOGGER.warning("No weight_constant provided in config, using default value of 1.0.")
                if self.problem_type == "TSP":
                    dummy_dict = self.generate_dummy_tsp(N, seed, weight_constant=weight_constant)
                else:
                    dummy_dict = self.generate_dummy_atsp(N, seed, weight_constant=weight_constant)
            elif self.problem_type == "MIMO":
                dummy_qam = self.config.dummy_qam if hasattr(self.config, "dummy_qam") else 4
                dummy_snr = self.config.dummy_snr if hasattr(self.config, "dummy_snr") else 10
                user_num = self.config.dummy_user_num if hasattr(self.config, "dummy_user_num") else 4
                ant_num = self.config.dummy_ant_num if hasattr(self.config, "dummy_ant_num") else 4
                dummy_dict = self.generate_dummy_mimo(user_num=user_num, ant_num=ant_num,
                                                      M=dummy_qam, SNR=dummy_snr, seed=seed,
                                                      dummy_trails=self.config.dummy_trails)
            elif self.problem_type == "Knapsack":
                N = self.config.dummy_size
                density = self.config.dummy_density if hasattr(self.config, "dummy_density") else 1
                penalty_value = self.config.dummy_penalty_value if hasattr(self.config, "dummy_penalty_value") else 1.0
                bit_width = self.config.dummy_bit_width if hasattr(self.config, "dummy_bit_width") else 16
                dummy_dict = self.generate_dummy_knapsack(size=N,
                                                          dns=density,
                                                          penalty_value=penalty_value,
                                                          bit_width=bit_width)
            else:
                LOGGER.error(f"Dummy creator for {self.problem_type} is not supported.")
                raise NotImplementedError(f"Dummy creator for {self.problem_type} is not implemented.")

            self.kwargs["config"] = self.config
            self.kwargs["dummy_dict"] = dummy_dict
            self.kwargs["best_found"] = None
        else:
            LOGGER.info("Dummy creator is disabled, skipping dummy model creation.")
            self.kwargs["config"] = self.config

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def generate_dummy_maxcut(N: int, seed: int = 0) -> dict:
        """! Generates a random Max Cut Ising model.
        @param N: Number of nodes in the graph.
        @param seed: Random seed for reproducibility.

        @return dummy_dict: dict containing graph and IsingModel representing the Max Cut problem.
        """

        np.random.seed(seed)
        name = f"DummyMaxCut_N{N}_seed{seed}"
        J = np.random.choice([-0.5, 0.0, 0.5], (N, N), p=[0.15, 0.7, 0.15])

        # Map the J matrix to a graph
        graph = nx.Graph(name=name)
        graph.add_nodes_from(range(1, N + 1))  # Nodes are 1-indexed in the graph
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    graph.add_edge(i + 1, j + 1, weight=-J[i, j] * 2)

        J = np.triu(J, k=1)  # Keep only upper triangle
        h = np.zeros((N,))  # No external field
        c = np.sum(J)  # Constant term for the Max Cut problem
        ising_model = IsingModel(J, h, c, name=name)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_tsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> dict:
        """! Generates a random TSP Ising model.
        @param N: Number of cities (nodes) in the TSP problem.
        @param seed: Random seed for reproducibility.
        @param weight_constant: Constant to scale the weights in the TSP problem.

        @return dummy_dict: dict containing graph and IsingModel representing the TSP problem.
        """

        np.random.seed(seed)
        name = f"DummyTSP_N{N}_seed{seed}"
        W = np.random.choice(10, (N, N))
        W = (W + W.T) / 2  # Make it symmetric

        graph = nx.DiGraph(name=name)
        graph.add_nodes_from(range(1, N + 1))
        for i in range(N):
            for j in range(N):
                if i != j:
                    if W[i, j] != 0:
                        graph.add_edge(i + 1, j + 1, weight=W[i, j])

        ising_model = TSP(graph, weight_constant=weight_constant)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
            "weight_constant": weight_constant,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_atsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> dict:
        """! Generates a random ATSP Ising model.
        @param N: Number of cities (nodes) in the ATSP problem.
        @param seed: Random seed for reproducibility.
        @param weight_constant: Constant to scale the weights in the ATSP problem.

        @return dummy_dict: dict containing graph and IsingModel representing the ATSP problem.
        """

        np.random.seed(seed)
        name = f"DummyATSP_N{N}_seed{seed}"
        W = np.random.choice(10, (N, N))

        graph = nx.DiGraph(name=name)
        graph.add_nodes_from(range(1, N + 1))
        for i in range(N):
            for j in range(N):
                if i != j:
                    if W[i, j] != 0:
                        graph.add_edge(i + 1, j + 1, weight=W[i, j])

        ising_model = TSP(graph, weight_constant=weight_constant)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
            "weight_constant": weight_constant,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_mimo(user_num:int, ant_num:int, M: int, SNR: int, seed: int = 0, dummy_trails: int = 10) -> dict:
        """!Generates a MU-MIMO model using section IV-A of [this paper](https://arxiv.org/pdf/2002.02750).
        This is consecutively transformed into an Ising model.

        @param ant_num (int): The amount of users.
        @param user_num (int): The amount of antennas at the Base Station.
        @param M (int): the considered QAM scheme.
        @param SNR (int): the Signal-to-Noise Ratio.
        @param seed (int, optional): The seed for the random number generator. Defaults to 1.
        @param dummy_trails (int, optional): The number of dummy trails to generate. Defaults to 10.

        @return dummy_dict: dict containing IsingModel representing the MIMO problem.
        """
        np.random.seed(seed)

        if M==2:
            # BPSK scheme
            symbols = np.array([-1, 1])
            r = 1
        else:
            r = int(np.ceil(np.log2(np.sqrt(M))))
            symbols = np.concatenate(
                ([-np.sqrt(M) + i for i in range(1, 2 + 2 * r, 2)], [np.sqrt(M) - i for i in range(1, 2 + 2 * r, 2)])
            )

        phi_u     = 120 * (np.random.random((10, user_num)) - 0.5)
        phi_u.sort()
        mean_phi  = np.mean(phi_u, axis=0)
        sigma_phi = np.random.normal(0, 1, (user_num,))

        # H = np.random.random((ant_num, user_num)) + 1j*np.random.random((ant_num, user_num))
        H = np.zeros((ant_num, user_num), dtype='complex_')
        for i in range(user_num):
            C     = np.zeros((ant_num, ant_num), dtype="complex_")
            phi   = mean_phi[i]
            sigma = sigma_phi[i]
            for m in range(ant_num):
                for n in range(ant_num):
                    d = np.abs(m-n)
                    C[m, n] = np.exp(2*np.pi*1j*d*np.sin(phi))* np.exp(
                        -(sigma**2) / 2 * (2 * np.pi * d * np.cos(phi)) ** 2
                    )
            D, V = np.linalg.eig(C)
            hu = V @ np.diag(D)**0.5 @ V.conj().T @ (
                np.random.normal(0, 1, (ant_num,)) + 1j*np.random.normal(0, 1, (ant_num,)))
            H[:, i] = hu
        x_collect: list = []
        for i in range(dummy_trails):
            x = np.random.choice(symbols, size=(user_num,)) + 1j * np.random.choice(symbols, size=(user_num,))
            x_collect.append(x)
        x_collect: np.ndarray = np.array(x_collect).T  # shape (user_num, dummy_trails)
        dummy_dict: dict = {
            "H": H,
            "x_collect": x_collect,
            "user_num": user_num,
            "ant_num": ant_num,
            "M": M,
            "SNR": SNR,
            "seed": seed,
        }
        return dummy_dict

    def generate_dummy_knapsack(size: int, dens: int, penalty_value: float = 1.0, bit_width: int = 16) -> dict:
        """! Generates a dummy knapsack problem instance.

        @param size (int): the number of items.
        @param dens (int): the density of the problem.
        @param penalty_value (float, optional): the penalty value for the constraint. Defaults to 1.0.

        @return IsingModel: the corresponding Ising model.
        """
        max_number = int(2**bit_width)
        profit = np.triu(
            np.random.choice(
                max_number + 1, size=(size, size), p=[1, -dens / 100] + [dens / (dens * max_number)] * (max_number - 1)
            )
        )
        profit = profit + profit.T
        weights = np.random.randint(1, max_number, size=(size,))
        capacity = np.random.randint(np.min(weights) * 2, np.sum(weights) - np.min(weights), size=(1,))[0]

        ising_model: IsingModel = QKPParserStage([StageCallable], config=Namespace()).knapsack_to_ising(
            profit, capacity, weights, penalty_value
        )

        dummy_dict: dict = {
            "ising_model": ising_model,
            "N": size,
            "density": dens,
            "penalty_value": penalty_value,
            "bit_width": bit_width,
        }

        return dummy_dict
