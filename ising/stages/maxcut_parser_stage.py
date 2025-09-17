from ising.stages import LOGGER, TOP
from typing import Any
import networkx as nx
import numpy as np
import pathlib
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

class MaxcutParserStage(Stage):
    """! Stage to parse the Maxcut benchmark workload."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / config.benchmark

    def run(self) -> Any:
        """! Parse the Maxcut benchmark workload."""

        if self.config.dummy_creator:
            dummy_dict = self.kwargs.get("dummy_dict", {})
            graph = dummy_dict.get("graph", None)
            best_found = dummy_dict.get("best_found", None)
            ising_model = dummy_dict.get("ising_model", None)
        else:
            LOGGER.debug(f"Parsing Maxcut benchmark: {self.benchmark_filename}")
            graph: nx.Graph
            best_found: float | None
            graph, best_found = self.G_parser(benchmark=self.benchmark_filename)
            ising_model: IsingModel = self.generate_maxcut(graph=graph)

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = ising_model
        self.kwargs["nx_graph"] = graph
        self.kwargs["best_found"] = best_found
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def generate_maxcut(graph: nx.Graph) -> IsingModel:
        """! Generates an Ising model from the given undirected nx graph

        @param graph (nx.Graph): graph on which the max-cut problem will be solved

        @return model (IsingModel): generated model from the graph
        """
        N = len(graph.nodes)
        coupling = -nx.adjacency_matrix(graph, weight='weight').toarray()/2
        coupling = np.triu(coupling)
        h = np.zeros((N,))
        c = np.sum(coupling)
        return IsingModel(coupling, h, c, name=graph.name)

    @staticmethod
    def G_parser(benchmark: pathlib.Path | str) -> tuple[nx.DiGraph, float]:
        """! Creates undirected graph from G benchmark.

        @param benchmark: benchmark that needs to be generated.

        @return G: a tuple containing the graph and best found cut value.
        @return best_found: the best found cut value.
        """
        data = False
        name = str(benchmark).split("/")[-1].split(".")[0]
        G = nx.Graph(name=name)
        with benchmark.open() as f:
            for line in f:
                if not data:
                    row = line.split()
                    N = int(row[0])
                    G.add_nodes_from(list(range(N)))
                    data = True
                else:
                    line = line.split()
                    u = int(line[0]) - 1
                    v = int(line[1]) - 1
                    weight = float(line[2])
                    G.add_edge(u, v, weight=weight)

        best_found = MaxcutParserStage.get_optim_value(benchmark)
        return G, best_found

    def get_optim_value(benchmark: pathlib.Path | str) -> float | None:
        """! Returns the best found value of the benchmark if the optimal value is known.

        @param benchmark: the benchmark file

        @return: best_found: the best found energy of the benchmark
        """
        best_found = None
        benchmark_name = str(benchmark).split("/")[-1]
        if benchmark_name.endswith(".txt"):
            benchmark_name = benchmark_name[:-4]
        elif benchmark_name.endswith(".sparse"):
            benchmark_name = benchmark_name[:-7]
        else:
            pass
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
