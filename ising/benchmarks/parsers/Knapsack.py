import numpy as np
import pathlib

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def QKP_parser(benchmark:pathlib.Path | str) -> tuple[np.ndarray, np.ndarray, float, float]:

    profit_part = False
    capacity_part = False
    weight_part = False

    capacity = None
    profit = None
    first_profit_line = False
    weights = None
    N = 0
    with benchmark.open() as file:
        for line in file:
            if profit_part:
                if first_profit_line:
                    parts = np.array(line.split(), dtype=int)
                    profit = np.diag(parts)
                    first_profit_line = False
                    i = 0
                elif i < N-1:
                    parts = np.array(line.split(), dtype=int)
                    profit[i, i+1:] = parts
                    profit[i+1:, i] = parts
                    i += 1
                else:
                    profit_part = False

            elif weight_part:
                parts = line.split()
                weights = np.array(parts, dtype=int)

            elif capacity_part:
                capacity = int(line)
                capacity_part = False
                weight_part = True

            else:
                parts = line.split("_")
                if len(parts) == 4:
                    N = int(parts[1])
                elif int(parts[0]) == N:
                    profit_part = True
                    first_profit_line = True
                elif int(parts[0]) == 0:
                    capacity_part = True
    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/Knapsack/optimal_energy.txt")
    return profit, weights, capacity, best_found



