from ising.stages import LOGGER, TOP
from typing import Any
import networkx as nx
import numpy as np
import pathlib
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel
from ising.generators.Knapsack import knapsack


class QKPParserStage(Stage):
    """! Stage to parse the QKP benchmark workload."""

    def __init__(self, list_of_callables: list[StageCallable], *, config: Any, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / config.benchmark

    def run(self) -> Any:
        """! Parse the Knapsack benchmark workload."""

        LOGGER.debug(f"Parsing Knapsack benchmark: {self.benchmark_filename}")
        graph: nx.Graph
        best_found: float | None
        graph, best_found = self.QKP_parser(benchmark=self.benchmark_filename)

        penalty_value = float(self.config.penalty_value)
        ising_model: IsingModel = self.generate_knapsack(graph=graph, penalty_value=penalty_value)

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = ising_model
        self.kwargs["nx_graph"] = graph
        self.kwargs["best_found"] = best_found
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def generate_knapsack(graph: nx.Graph, penalty_value: float) -> IsingModel:
        """! Generates an Ising model from the given undirected nx graph

        @param graph (nx.Graph): graph on which the knapsack problem will be solved

        @return model (IsingModel): generated model from the graph
        """
        N = len(graph.nodes)
        profit_edges = nx.get_edge_attributes(graph, "profit")
        profit = np.zeros((N, N))
        for (i, j), value in profit_edges.items():
            profit[i, j] = value
            profit[j, i] = value
        weight_edges = nx.get_edge_attributes(graph, "weight")
        weights = np.array([weight_edges[i] for i in range(N)])
        capacity = graph.graph["capacity"]
        return knapsack(profit, capacity, weights, penalty_value)

    @staticmethod
    def QKP_parser(benchmark: pathlib.Path | str) -> tuple[nx.DiGraph, float]:
        """! Creates undirected graph from QKP benchmark.

        @param benchmark: benchmark that needs to be generated.

        @return G: a networkx object containing the all the data of the benchmark.
        @return best_found: the best found cut value.
        """
        # Make sure we keep track of where we are in the file
        profit_part = False
        capacity_part = False
        weight_part = False

        # Initialize the variables
        capacity = None
        profit = None
        first_profit_line = False
        weights = None
        N = 0
        with benchmark.open() as file:
            for line in file:
                if profit_part:
                    if first_profit_line:
                        # First profit line holds the diagonal values of the matrix
                        parts = np.array(line.split(), dtype=int)
                        profit = np.diag(parts)
                        first_profit_line = False
                        i = 0
                    elif i < N - 1:
                        # The rest is stored in an upper triangular way. But we need to make it symmetric.
                        parts = np.array(line.split(), dtype=int)
                        profit[i, i + 1 :] = parts
                        profit[i + 1 :, i] = parts
                        i += 1
                    else:
                        # Profit part is done, set value to False.
                        profit_part = False

                elif weight_part:
                    # Weight are stored as a single line of weights.
                    parts = line.split()
                    weights = np.array(parts, dtype=int)
                    weight_part = False

                elif capacity_part:
                    # Capacity is stored as single integer on a line.
                    # Just afterwards, the weights are stored.
                    capacity = int(line)
                    capacity_part = False
                    weight_part = True

                else:
                    # No special line is present, so checking for the start of a new part.
                    parts = line.split("_")
                    if len(parts) == 4:
                        # Very first line holds the data about what kind of problem it is.
                        # Extract the data of the size from it.
                        N = int(parts[1])
                    elif parts[0] == str(N) + "\n":
                        # Second line holds the number of items and the following line holds the profit matrix.
                        # Thus we set the profit part to True.
                        profit_part = True
                        first_profit_line = True
                    elif parts[0] == str(0) + "\n":
                        # After the profit matrix there is an empty line. Use this to set the capacity part to True.
                        capacity_part = True
        best_found = -QKPParserStage.get_optim_value(benchmark)

        G = nx.Graph(capacity=capacity)
        G.add_nodes_from(range(N))
        G.add_weighted_edges_from(
            ((i, j, profit[i, j]) for i in range(N) for j in range(i, N) if profit[i, j] != 0.0), weight="profit"
        )
        G.add_weighted_edges_from(((i, i, weights[i]) for i in range(N)), weight="weight")
        return G, best_found

    def get_optim_value(benchmark: pathlib.Path | str) -> float | None:
        """! Returns the best found value of the benchmark if the optimal value is known.

        @param benchmark: the benchmark file

        @return: best_found: the best found energy of the benchmark
        """
        best_found = None
        benchmark_name = str(benchmark).split("/")[-1][:-4]

        benchmark_parent_folder = pathlib.Path(benchmark).parent
        optim_file = benchmark_parent_folder / "optimal_energy.txt"
        if not optim_file.exists():
            LOGGER.warning(f"Optimal energy file {optim_file} does not exist. Returning None.")
        else:
            with optim_file.open() as f:
                for line in f:
                    line = line.split()
                    if line[0] == benchmark_name:
                        best_found = -float(line[1])
                        break

        return best_found
