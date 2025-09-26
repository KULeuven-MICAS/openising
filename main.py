import logging
import sys
from pathlib import Path
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import numpy as np
from ising import api


# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "MIMO"  # Specify the problem type
config_path = "ising/inputs/config/config_pseudo_random.yaml"

# Run the Ising model simulation
ans, debug_info = api.get_hamiltonian_energy(
    problem_type=problem_type,
    config_path=config_path,
    logging_level=logging_level,
)

# Output summary file
output_file = Path("./simulation_summary.pkl")
solvers = ans.config.solvers
mean_computation_time = {solver: np.mean(ans.computation_time[solver]) for solver in solvers}
comp_str = " ".join([f"{mean_computation_time[solver]:.4f}s" for solver in solvers])
solver_str = " ".join(solvers)
operation_str = " ".join([f"{ans.operation_count[solver]}" for solver in solvers])
if problem_type == "MIMO":
    logging.info("BER: %s", ans.BER)
    BER_str = " ".join([str(ans.BER[solver]) for solver in solvers])
    with Path.open(output_file, "a") as f:
        f.write("\n")
        f.write("=====================\n")
        f.write(f"results of running {ans.benchmark} with {config_path.split('/')[-1]}:\n")
        f.write("=====================\n")
        f.write("MIMO results:\n")
        f.write(f"SNR|BER  {solver_str}\n")
        f.write(f"{ans.SNR}|     {BER_str}\n")
        f.write("\n")
        f.write("=====================\n")
        f.write("Simulation results:\n")
        f.write(f"solver| {solver_str}\n")
        f.write(f"computation time| {comp_str}\n")
        f.write(f"operation count| {operation_str}\n")
else:
    benchmark = ans.benchmark
    ising_energies = ans.energies
    best_found = ans.best_found
    ising_energy_max = {solver: np.max(ising_energies[solver]) for solver in solvers}
    max_en_str = " ".join([f"{ising_energy_max[solver]:.4f}" for solver in solvers])
    ising_energy_min = {solver: np.min(ising_energies[solver]) for solver in solvers}
    min_en_str = " ".join([f"{ising_energy_min[solver]:.4f}" for solver in solvers])
    ising_energy_avg = {solver: np.mean(ising_energies[solver]) for solver in solvers}
    avg_en_str = " ".join([f"{ising_energy_avg[solver]:.4f}" for solver in solvers])
    tts = {
        solver: mean_computation_time[solver]
        + np.log(1 - 0.99) / np.log(1 - (np.sum(ising_energies[solver] == 0.98 * best_found) / ans.config.nb_runs))
        for solver in solvers
    }
    approximation = {solver: 100 * ising_energy_avg[solver] / best_found for solver in solvers}
    approx_str = " ".join([f"{approximation[solver]:.2f}%" for solver in solvers])
    tts_str = " ".join([f"{tts[solver]:.4f}s" for solver in solvers])
    with Path.open(output_file, "a") as f:
        f.write("\n")
        f.write("=====================\n")
        f.write(f"results of running {ans.benchmark} with {config_path.split('/')[-1]}:\n")
        f.write(f"reference energy {best_found}\n")
        f.write("=====================\n")
        f.write("Simulation results:\n")
        f.write(f"solver| {solver_str}\n")
        f.write(f"energy max| {max_en_str}\n")
        f.write(f"energy min| {min_en_str}\n")
        f.write(f"energy avg| {avg_en_str}\n")
        f.write(f"computation time| {comp_str}\n")
        f.write(f"TTT 0.98| {tts_str}\n")
        f.write(f"operation count| {operation_str}\n")
        f.write(f"approximation| {approx_str}\n")
        f.write("\n")

    logging.info(
        "benchmark: %s, reference: %s, energy max: %s, min: %s, avg: %s",
        benchmark,
        best_found,
        ising_energy_max,
        ising_energy_min,
        ising_energy_avg,
    )
