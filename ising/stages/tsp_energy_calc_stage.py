from ising.stages import LOGGER
from typing import Any
import numpy as np
import pathlib
import networkx as nx
import datetime

from ising.stages.stage import Stage, StageCallable
from ising.utils.HDF5Logger import HDF5Logger, return_data, return_metadata
from ising.generators.TSP import get_TSP_value

class TSPEnergyCalcStage(Stage):
    """! Stage to calculate the TSP energy for every state of the given logfiles."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 nx_graph: nx.DiGraph,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.use_gurobi = config.use_gurobi
        self.nx_graph = nx_graph

    def run(self) -> Any:
        """! Calculate the TSP energy for every state of the given logfiles."""

        self.kwargs["config"] = self.config
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            # Check if the answer is valid
            tsp_energies = {solver: [] for solver in self.config.solvers}
            for solver in self.config.solvers:
                for state_id in range(len(ans.states[solver])):
                    state = ans.states[solver][state_id]
                    city_count = self.nx_graph.number_of_nodes()
                    state = state.reshape(city_count, city_count)
                    binary_state = state.copy()
                    binary_state[binary_state == -1] = 0  # Convert -1 to 0 for TSP calculation
                    if np.linalg.norm(binary_state.T @ binary_state - np.eye(city_count)) == 0:
                        # Valid state
                        ising_energy = ans.energies[solver][state_id]
                        ## Uncomment the following lines if you want to calculate TSP energy from logfiles
                        # logfile = ans.logfiles[state_id]
                        # tsp_energy = self.calculate_TSP_energy(
                        #     logfile=logfile,
                        #     graph=self.nx_graph,
                        #     gurobi=self.use_gurobi,
                        # )
                        tsp_energies[solver].append(ising_energy) # ising_energy equals to the TSP cost
                    else:
                        # Invalid state, append NaN
                        tsp_energies[solver].append(np.nan)
            ans.tsp_energies = tsp_energies

            yield ans, debug_info

    @staticmethod
    def calculate_TSP_energy(logfile:pathlib.Path, graph:nx.DiGraph, gurobi:bool=False) -> float:
        """! Calculates the TSP distance for every state of the given logfiles.
        It will append this data to the file.

        @param logfile: the logfiles.
        @param graph: the original graph on which the TSP problem is solved. All the logfiles solved this problem.
        @param gurobi: whether the logfiles contain Gurobi data. Defaults to False.

        @return TSP_value: the TSP distance value.
        """
        start_time = datetime.datetime.now()
        LOGGER.info(f"TSP energy inferring started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        schema = {"TSP_energy": np.float64}
        if not gurobi:
            num_iterations = return_metadata(logfile, "num_iterations")
            samples = return_data(logfile, "state")
            with HDF5Logger(logfile, schema, mode="a") as logger:
                for i in range(num_iterations):
                    sample = samples[i, :]
                    TSP_value = get_TSP_value(graph, sample)
                    logger.log(TSP_energy=TSP_value)
                logger.write_metadata(solution_TSP_energy=TSP_value)
        else:
            solution_state = return_metadata(logfile, "solution_state")
            solution_state[solution_state==0] = -1
            TSP_value = get_TSP_value(graph, solution_state)
            with HDF5Logger(logfile, schema, mode='a') as logger:
                logger.write_metadata(solution_TSP_energy=TSP_value)
        end_time = datetime.datetime.now()
        LOGGER.info(f"TSP energy inferring finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.info(f"Total inferring time: {end_time - start_time}")
        return TSP_value
