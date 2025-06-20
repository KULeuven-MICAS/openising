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
            logfiles = ans.logfiles
            tsp_energies = self.calculate_TSP_energy(
                logfiles=logfiles,
                graph=self.nx_graph,
                gurobi=self.use_gurobi,
            )
            ans.tsp_energies = tsp_energies

            yield ans, debug_info

    @staticmethod
    def calculate_TSP_energy(logfiles:list[pathlib.Path], graph:nx.DiGraph, gurobi:bool=False) -> list[float]:
        """! Calculates the TSP energy for every state of the given logfiles.
        It will append this data to the file.

        @param logfiles: list of all the logfiles.
        @param graph: the original graph on which the TSP problem is solved. All the logfiles solved this problem.
        @param gurobi: whether the logfiles contain Gurobi data. Defaults to False.
        """
        start_time = datetime.datetime.now()
        LOGGER.info(f"TSP energy inferring started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        tsp_energy_collect = []
        for logfile in logfiles:
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
            tsp_energy_collect.append(TSP_value)
        end_time = datetime.datetime.now()
        LOGGER.info(f"TSP energy inferring finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.info(f"Total inferring time: {end_time - start_time}")
        return tsp_energy_collect
