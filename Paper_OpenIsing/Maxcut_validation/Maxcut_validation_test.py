import logging
import sys
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from ising import api
from ising.postprocessing.run_summary import summarize_runs
from ising.postprocessing.energy_plot import plot_energy_time_multiple, plot_energies_multiple
from ising.stages import TOP
import yaml
from ising.utils.parser import get_optim_value
from Paper_OpenIsing import BASE_PATH

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO]
simulation = False
base_path = BASE_PATH / "Maxcut_validation"
config_path = base_path / "K2000_experiment.yaml"
fig_path = base_path / "figures"
if simulation:
    # Run the Ising model simulation
    ans, debug_info = api.get_hamiltonian_energy(
        problem_type=problem_type,
        config_path=config_path.relative_to(TOP),
        logging_level=logging_level,
    )

    # Output summary file
    output_file = base_path / f"simulation_summary_{ans.benchmark}.txt"
    summarize_runs(output_file, ans, problem_type, config_path)


    plot_energies_multiple(
        ans.logfiles,
        figName=f"Maxcut_validation_{ans.benchmark}_energy_plot",
        best_found=ans.best_found,
        save_folder=fig_path,
    )

    plot_energy_time_multiple(
        ans.logfiles,
        ans.best_found,
        save_folder=fig_path,
        figName=f"Maxcut_validation_{ans.benchmark}_energy_time_plot",
    )
else:
    logfiles = []
    base_path = TOP / "ising/outputs/Maxcut/logs"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    solvers = list(config["solvers"])
    benchmark = config["benchmark"].split("/")[-1].split(".")[0]
    nb_trials = int(config["nb_runs"])
    for solver in solvers:
        for trial in range(nb_trials):
            logfile = base_path / f"{solver}_{benchmark}_run{trial}.log"
            logfiles.append(logfile)
    best_found = -get_optim_value(config["benchmark"], TOP / "ising/benchmarks/G/optimal_energy.txt")

    plot_energies_multiple(
        logfiles,
        figName=f"Maxcut_validation_{benchmark}_energy_plot",
        best_found=best_found,
        save_folder=fig_path,
    )

    plot_energy_time_multiple(
        logfiles,
        best_found,
        save_folder=fig_path,
        figName=f"Maxcut_validation_{benchmark}_energy_time_plot",
    )

