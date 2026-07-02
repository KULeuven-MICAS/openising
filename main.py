import logging
import sys
from pathlib import Path
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from ising import api
from ising.postprocessing.run_summary import summarize_runs
from ising.postprocessing.mppi_summary import summarize_mppi


# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, MPPI]
config_path = "ising/inputs/config/example_mppi.yaml"

# Run the Ising model simulation
ans, debug_info = api.get_hamiltonian_energy(
    problem_type=problem_type,
    config_path=config_path,
    logging_level=logging_level,
)

# Output summary file
output_file = Path(f"./simulation_summary_{ans.benchmark}.pkl")
summarize_runs(output_file, ans, problem_type, config_path)
summarize_mppi(Path("./"), ans)
