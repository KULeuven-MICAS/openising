import numpy as np
import pathlib

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def QKP_parser(benchmark:pathlib.Path | str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Parses a Knapsack benchmark file. The given file should have the same structure as the ones [here](https://cedric.cnam.fr/~soutif/QKP/QKP.html).

    Args:
        benchmark (pathlib.Path | str): The given benchmark file to parse.

    Returns:
        profit,weight,capacity,best_found (tuple[np.ndarray, np.ndarray, float, float]): the profit matrix, the weights
                                        vector, the capacity, and the best found energy solution.
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
                elif i < N-1:
                    # The rest is stored in an upper triangular way. But we need to make it symmetric.
                    parts = np.array(line.split(), dtype=int)
                    profit[i, i+1:] = parts
                    profit[i+1:, i] = parts
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
                    # Second line holds the number of items and the following line holds the profit matrix. Thus we set
                    # the profit part to True.
                    profit_part = True
                    first_profit_line = True
                elif parts[0] == str(0) + "\n":
                    # After the profit matrix there is an empty line. Use this to set the capacity part to True.
                    capacity_part = True
    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/Knapsack/optimal_energy.txt")
    return profit, weights, capacity, best_found



