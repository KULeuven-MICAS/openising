import numpy as np
import pathlib
import networkx as nx

from ising.utils.HDF5Logger import HDF5Logger, return_data, return_metadata
from ising.generators.TSP import get_TSP_value

def calculate_TSP_energy(logfiles:list[pathlib.Path], graph:nx.DiGraph):
    """Calculates the TSP energy for every state of the given logfiles.
    It will append this data to the file.

    Args:
        logfiles (list[pathlib.Path]): list of all the logfiles.
        graph (nx.DiGraph): the original graph on which the TSP problem is solved. All the logfiles solved this problem.
    """
    for logfile in logfiles:
        schema = {"TSP_value": np.float64}
        num_iterations = return_metadata(logfile, "num_iterations")
        samples = return_data(logfile, "state")
        with HDF5Logger(logfile, schema, mode="a") as logger:
            for i in range(num_iterations):
                sample = samples[i, :]
                TSP_value = get_TSP_value(graph, sample)
                logger.log(TSP_value=TSP_value)
