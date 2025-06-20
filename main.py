import logging
import sys
import numpy as np
from ising import api

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type
config_path = "ising/inputs/config/config_maxcut.yaml"

# Run the Ising model simulation
ans, debug_info = api.get_hamiltonian_energy(
    problem_type=problem_type,
    config_path=config_path,
    logging_level=logging_level,
)
benchmark = ans.benchmark
ising_energies = ans.energies
ising_energy_max = np.max(ising_energies)
ising_energy_min = np.min(ising_energies)
ising_energy_avg = np.mean(ising_energies)

logging.info(
    f"bemchmark: {benchmark}, energy max: {ising_energy_max}, min: {ising_energy_min}, avg: {ising_energy_avg}")
