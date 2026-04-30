from ising.workloads.run_workload import run_workload
from ising.stages import TOP
from ising.postprocessing.run_summary import summarize_workload
import yaml
"""
Choose type of the Galena solver to be used. Options are "base", "HW", "comb_nodes_HW", "multi_core", and
"comb_nodes-multi_core".

- "base": the base galena solver without any HW assumptions made.
- "HW": this options adds all HW assumptions to the solver. These include quantization of the interaction weights J,
    mismatch on the interaction weights, and delay. All sub-assumptions can be turned on or
    off in the config file.
- "comb_nodes_HW": this option adds the first solver improvement to the solver, given the HW assumptions.
    The improvement allows one node to be represented by a combination of nodes
    to increase quantization precision.
- "multi_core": this option adds the second solver improvement to the solver, given the HW assumptions.
    The improvement allows the solver to split the model into multiple cores to increase the amount of nodes
    that can be handled. Currently not implemented.
- "comb_nodes-multi_core": this option adds both the first and second solver improvements to the solver.

"""
top_benchmark = "./ising/benchmarks/TSP/"
solver_type = "base"
config_file = "./ising/inputs/config/config_tspWorkload.yaml"
difficulty = "easy"  # easy - < 25 cities, medium - 25 <= cities <= 40, difficult - > 40 cities

settings = {
    "current": 1e-6,
    "capacitance": 1e-15,
    "quantization": True,
    "quantization_precision": 4,
    "mismatch_std": 0.1,
    "sigma_J": 0.0,
    "accumulation_delay": 0,
    "broadcast_delay": 0,
    "delay_offset": 0,
    "nodes_scaling": 2,
    "nb_cores": 2,
}

if difficulty == "easy":
    problems = ["burma14.tsp", "gr17.tsp", "ulysses16.tsp", "gr21.tsp", "gr24.tsp", "ulysses22.tsp"]
elif difficulty == "medium":
    problem = ["bayg29.tsp", "bays29.tsp", "fri26.tsp"]
elif difficulty == "difficult":
    problem = ["att48.tsp", "berlin52.tsp", "brazil58.tsp", "dantzig42.tsp", "gr48.tsp", "hk48.tsp"]
else:
    raise ValueError("Invalid difficulty level. Options are 'easy', 'medium', or 'difficult'.")

ans_list = []
for problem in problems:
    with (TOP / config_file).open("r") as f:
        config = yaml.safe_load(f)
    config["benchmark"] = top_benchmark + problem
    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)
    ans, _ = run_workload(problem_type="TSP", solver_type=solver_type, config_file=config_file, **settings)
    ans_list.append(ans)
summarize_workload(
    output_file=TOP / f"ising/workloads/tsp_results_{difficulty}_{solver_type}.txt",
    problem_type="TSP",
    config_path=TOP / config_file,
    ans_list=ans_list,
)
