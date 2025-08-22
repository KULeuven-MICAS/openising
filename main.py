import logging
import sys
import numpy as np
from ising import api
import os

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

os.system("rm -rf ising/outputs/TSP/logs/*")  # Clear previous logs

# Input file directory
problem_type = "MIMO"  # Specify the problem type
config_path = "ising/inputs/config/example.yaml"

# Run the Ising model simulation
ans, debug_info = api.get_hamiltonian_energy(
    problem_type=problem_type,
    config_path=config_path,
    logging_level=logging_level,
)
if problem_type == "MIMO":
    logging.info("BER: %s", ans.BER)
else:
    benchmark = ans.benchmark
    ising_energies = ans.energies
    best_found = ans.best_found
    ising_energy_max = np.max(ising_energies)
    ising_energy_min = np.min(ising_energies)
    ising_energy_avg = np.mean(ising_energies)

    logging.info(
        "benchmark: %s, reference: %s, energy max: %s, min: %s, avg: %s",
        benchmark,
        best_found,
        ising_energy_max,
        ising_energy_min,
        ising_energy_avg
    )
