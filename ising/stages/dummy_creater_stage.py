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
        N = self.config.dummy_size
        seed = self.config.seed

        if self.problem_type == "Maxcut":
            graph, ising_model = self.generate_dummy_maxcut(N, seed)
        elif self.problem_type in ["TSP", "ATSP"]:
            weight_constant = self.config.weight_constant if hasattr(self.config, "weight_constant") else 1.0
            if not hasattr(self.config, "weight_constant"):
                LOGGER.warning("No weight_constant provided in config, using default value of 1.0.")
            if self.problem_type == "TSP":
                graph, ising_model = self.generate_dummy_tsp(N, seed, weight_constant=weight_constant)
            else:
                graph, ising_model = self.generate_dummy_atsp(N, seed, weight_constant=weight_constant)
        elif self.problem_type == "MIMO":
            graph, ising_model = self.generate_dummy_mimo(N, seed)
        else:
            LOGGER.error(f"Dummy creator for {self.problem_type} is not supported.")
            raise NotImplementedError(f"Dummy creator for {self.problem_type} is not implemented.")

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = ising_model
        self.kwargs["nx_graph"] = graph
        self.kwargs["best_found"] = None

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def generate_dummy_maxcut(N: int, seed: int = 0) -> tuple[nx.DiGraph, IsingModel]:
        """! Generates a random Max Cut Ising model.
        @param N: Number of nodes in the graph.
        @param seed: Random seed for reproducibility.

        @return graph: A NetworkX graph representing the Max Cut problem.
        @return ising_model: An IsingModel object representing the Max Cut problem.
        """

        np.random.seed(seed)
        name = f"DummyMaxCut_N{N}_seed{seed}"
        J = np.random.choice([-0.5, 0.0, 0.5], (N, N), p=[0.15, 0.7, 0.15])

        # Map the J matrix to a graph
        graph = nx.Graph(name=name)
        graph.add_nodes_from(1, range(N + 1))  # Nodes are 1-indexed in the graph
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    graph.add_edge(i + 1, j + 1, weight=-J[i, j] * 2)

        J = np.triu(J, k=1)  # Keep only upper triangle
        h = np.zeros((N,))  # No external field
        c = np.sum(J)  # Constant term for the Max Cut problem
        ising_model = IsingModel(J, h, c, name=name)

        return graph, ising_model

    @staticmethod
    def generate_dummy_tsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> tuple[nx.DiGraph, IsingModel]:
        """! Generates a random TSP Ising model.
        @param N: Number of cities (nodes) in the TSP problem.
        @param seed: Random seed for reproducibility.
        @param weight_constant: Constant to scale the weights in the TSP problem.

        @return graph: A NetworkX directed graph representing the TSP problem.
        @return ising_model: An IsingModel object representing the TSP problem.
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

        return graph, ising_model

    @staticmethod
    def generate_dummy_atsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> tuple[nx.DiGraph, IsingModel]:
        """! Generates a random ATSP Ising model.
        @param N: Number of cities (nodes) in the ATSP problem.
        @param seed: Random seed for reproducibility.
        @param weight_constant: Constant to scale the weights in the ATSP problem.

        @return graph: A NetworkX directed graph representing the ATSP problem.
        @return ising_model: An IsingModel object representing the ATSP problem.
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

        return graph, ising_model

    @staticmethod
    def generate_dummy_mimo(N: int, seed: int = 0) -> tuple[nx.Graph, IsingModel]:
        """! Generates a random MIMO Ising model.
        @param N: Number of nodes in the MIMO problem.
        @param seed: Random seed for reproducibility.

        @return graph: A NetworkX graph representing the MIMO problem.
        @return ising_model: An IsingModel object representing the MIMO problem.
        """
        # Placeholder for MIMO generation logic
        raise NotImplementedError("Dummy creator for MIMO is not implemented.")

    def generate_dummy_knapsack(size: int, dens: int, penalty_value: float = 1.0, bit_width: int = 16) -> IsingModel:
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

        return QKPParserStage([StageCallable], config=Namespace()).knapsack_to_ising(
            profit, capacity, weights, penalty_value
        )
