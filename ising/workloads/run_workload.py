import yaml
from pathlib import Path

from ising.api import get_hamiltonian_energy
from ising.stages import TOP

def run_workload(problem_type, solver_type, config_file, **kwargs):

    """
    Runs the specified workload with the given solver type and configuration file.

    Args:
        problem_type (str): The type of problem to solve (e.g., "Maxcut", "TSP", etc.).
        solver_type (str): The type of Galena solver to use (e.g., "base", "HW", etc.).
        config_file (str): The path to the configuration file for the workload.

    Returns:
        ans: The answer obtained from running the workload.
        debug_info: Debug information collected during the run.
    """

    settings_def = {"current": kwargs["current"],
                    "capacitance": kwargs["capacitance"],}

    settings = dict()
    if solver_type in ["HW", 'comb_nodes_HW', "multi_core", "comb_nodes-multi_core"]:
        settings = {"quantization": kwargs["quantization"],
                    "quantization_precision": kwargs["quantization_precision"],
                    "mismatch_std": kwargs["mismatch_std"],
                    "sigma_J": kwargs["sigma_J"],
                    "accumulation_delay": kwargs["accumulation_delay"],
                    "broadcast_delay": kwargs["broadcast_delay"],
                    "delay_offset": kwargs["delay_offset"]}
        if solver_type == "comb_nodes_HW" or solver_type == "comb_nodes-multi_core":
            settings["combine_nodes"] = True
            settings["nodes_scaling"] = kwargs["nodes_scaling"]
        elif solver_type == "multi_core" or solver_type == "comb_nodes-multi_core":
            pass
            settings["multi_core"] = True
            settings["nb_cores"] = kwargs["nb_cores"]
    for param, val in settings_def.items():
        settings[param] = val

    with (TOP / config_file).open() as f:
        config = yaml.safe_load(f)
    for param, setting in settings.items():
        config[param] = setting

    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)

    # Run the workload using the API
    ans, debug_info = get_hamiltonian_energy(problem_type=problem_type, config_path=config_file)
    output_folder = TOP / f"ising/outputs/{problem_type}/ans"
    Path.mkdir(output_folder)
    ans.save(output_folder / f"{ans.benchmark}_{solver_type}.ans")
    return ans, debug_info
