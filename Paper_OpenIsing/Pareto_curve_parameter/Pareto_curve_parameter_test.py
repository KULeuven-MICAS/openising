import logging
import sys

from ising.stages import TOP, LOGGER
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from Paper_OpenIsing import BASE_PATH
from ising.stages.simulation_stage import Ans
from ising import api
from ising.postprocessing.summarize_energies import pareto_curve_loop

"""
This experiments tests the effect of changing a specific parameter of a solver on different problems.
The base yaml file configuration is loaded and modified for each parameter value. Then the configuration file
is used to run the simulation given the specific parameter value.

After all parameter values and problems are tested, the results are plotted in a pareto curve showcasing the relative
best found Ising energy for every problem.
"""

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
test_parameter = "dtSB"

problem_type = ["Maxcut", "Biqmac", "TSP", "QKP"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, QKP]
nb_benchmarks = 3
# Create folders for correct saving
save_folder_base = BASE_PATH / "Pareto_curve_parameter"

fig_folder = save_folder_base / "figures"
fig_folder.mkdir(parents=True, exist_ok=True)
data_folder = save_folder_base / "data"
data_folder.mkdir(parents=True, exist_ok=True)
config_folder = save_folder_base / "config"

# Load the original config file
config_path = config_folder / "Test_OPS_SB.yaml"

with config_path.open("r") as f:
    config = yaml.safe_load(f)

solvers = config["solvers"]
parameter_values = config[test_parameter]
ans_all = {
    problem: {param_val: [Ans() for _ in range(nb_benchmarks)] for param_val in parameter_values}
    for problem in problem_type
}

for i, problem in enumerate(problem_type):
    print("===============================")
    LOGGER.info(f"Starting simulation for {problem}...")

    # Run for all different parameters
    print("===============================")
    LOGGER.info(f"Testing parameter: {test_parameter}...")
    for benchmark in range(nb_benchmarks):
        # Set up new config file
        new_config = config.copy()
        new_config["benchmark"] = config["benchmark"][i * nb_benchmarks + benchmark]
        print("===============================")
        LOGGER.info(f"Testing benchmark: {config['benchmark'][i * nb_benchmarks + benchmark]}...")

        for value in parameter_values:
            if test_parameter == "dtSB":
                for solver in solvers:
                    new_config[f"dt{solver}"] = value
            else:
                new_config[test_parameter] = value

            # Make new config file
            config_file = config_folder / f"config_{problem}_{test_parameter}.yaml"
            with (config_file).open("w") as f:
                yaml.dump(new_config, f)

            data_file = data_folder / f"ans_{problem}_benchmark{benchmark}_{test_parameter}_{value}.pkl"
            if (data_file).exists():
                ans_all[problem][value][benchmark].load(data_file)
            else:
                ans, _ = api.get_hamiltonian_energy(
                    problem_type=problem, config_path=str(config_file.relative_to(TOP)), logging_level=logging_level
                )
                ans_all[problem][value][benchmark]=ans
                ans.save(data_file)
                LOGGER.info(f"Results stored in {data_file}")

        # Plot results
        parameter_name = test_parameter  # .replace("_", " ")
        for solver in solvers:
            if solver in parameter_name or solver[1:] in parameter_name:
                parameter_name = parameter_name.replace(solver, "")
                parameter_name = parameter_name.replace(solver[1:], "")
            if parameter_name[-1] == " ":
                parameter_name = parameter_name[:-1]


pareto_curve_loop(
    ans_all,
    parameter_name,
    parameter_values,
    problem_type,
    fig_folder,
    fig_name=f"Pareto_curve_parameter_{parameter_name}",
)
