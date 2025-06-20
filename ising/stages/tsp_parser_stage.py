from ising.stages import LOGGER, TOP
from typing import Any
import networkx as nx
import pathlib
import tsplib95
from ising.generators.TSP import TSP
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

class TSPParserStage(Stage):
    """! Stage to parse the TSP benchmark workload."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / config.benchmark

    def run(self) -> Any:
        """! Parse the TSP benchmark workload."""

        LOGGER.debug(f"Parsing TSP benchmark: {self.benchmark_filename}")
        graph: nx.Graph
        best_found: float | None
        graph, best_found = self.TSP_parser(benchmark=self.benchmark_filename)
        A = float(self.config.weight_constant)
        ising_model: IsingModel = TSP(graph=graph, weight_constant=A)

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = ising_model
        self.kwargs["nx_graph"] = graph
        self.kwargs["best_found"] = best_found
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def TSP_parser(benchmark:pathlib.Path)->tuple[nx.DiGraph, float]:
        """! Creates a graph from the given benchmark. With this graph a TSP problem can be generated.
        It is important to note that only TSP benchmarks can be used

        @param benchmark: the absolute path to the benchmark file.

        @return graph: a tuple containing the graph and best found energy.
        """
        if not benchmark.exists():
            LOGGER.error(f"Benchmark does not exist: {benchmark}")
            LOGGER.error("Returning None")
            return None
        benchmark = str(benchmark)
        name = benchmark.split("/")[-1].split(".")[0]
        problem = tsplib95.load(benchmark)
        graph = problem.get_graph()
        graph.name = name
        best_found = TSPParserStage.get_optim_value(benchmark)
        return graph, best_found

    @staticmethod
    def get_optim_value(benchmark:pathlib.Path|str)->float | None:
        """! Returns the best found value of the benchmark if the optimal value is known.

        @param benchmark: the benchmark file

        @return: best_found: the best found energy of the benchmark
        """
        benchmark = str(benchmark).split("/")[-1].split(".")[0]
        optim_file = TOP / "ising/benchmarks/TSP/optimal_energy.txt"
        best_found = None

        with optim_file.open() as f:
            for line in f:
                line = line.split()
                if line[0] == benchmark:
                    best_found = float(line[1])
                    break

        return best_found
