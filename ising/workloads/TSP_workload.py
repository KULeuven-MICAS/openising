from ising.workloads.run_workload import run_workload, SolverConfig
from ising.stages import TOP
from ising.postprocessing.run_summary import summarize_workload
from ising.postprocessing.summarize_energies import ans_to_metric_df, box_plot_metric
import yaml
"""
Pick which Galena solver features to enable via SolverConfig:
- SolverConfig() — base solver, no HW assumptions
- SolverConfig(hw=True) — HW assumptions on (quantized weights, mismatch, delay; toggle
    sub-assumptions in the config file)
- SolverConfig(hw=True, comb_nodes=True) — adds the combine-nodes improvement (one node
    represented by a combination of nodes to raise quantization precision)
- SolverConfig(hw=True, multi_core=True) — adds the multi-core improvement (splits the
    model across cores to scale node count). Currently not implemented.
- SolverConfig(hw=True, comb_nodes=True, multi_core=True) — both improvements.
"""
top_benchmark = "./ising/benchmarks/TSP/"
solver_config = SolverConfig()
config_file = "./ising/inputs/config/config_tspWorkload.yaml"
difficulty = "easy"  # easy - < 25 cities, medium - 25 <= cities <= 40, difficult - > 40 cities

if difficulty == "easy":
    problems = ["burma14.tsp", "gr17.tsp", "ulysses16.tsp", "gr21.tsp", "gr24.tsp", "ulysses22.tsp"]
elif difficulty == "medium":
    problems = ["bayg29.tsp", "bays29.tsp", "fri26.tsp"]
elif difficulty == "difficult":
    problems = ["att48.tsp", "berlin52.tsp", "brazil58.tsp", "dantzig42.tsp", "gr48.tsp", "hk48.tsp"]
else:
    raise ValueError("Invalid difficulty level. Options are 'easy', 'medium', or 'difficult'.")

ans_list = []
for problem in problems:
    with (TOP / config_file).open("r") as f:
        config = yaml.safe_load(f)
    config["benchmark"] = top_benchmark + problem
    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)
    ans, _ = run_workload(problem_type="TSP", solver_config=solver_config, config_file=config_file)
    ans_list.append(ans)
summarize_workload(
    output_file=TOP / f"ising/workloads/tsp_results_{difficulty}_{solver_config.tag}.txt",
    problem_type="TSP",
    config_path=TOP / config_file,
    ans_list=ans_list,
)

df = ans_to_metric_df({"all": ans_list}, label_name="bucket", problem="TSP")
box_plot_metric(
    df,
    x="benchmark",
    problem="TSP",
    title=f"TSP - {difficulty} difficulty, {solver_config.tag} solver",
    save_path=TOP / f"ising/workloads/tsp_boxplot_{difficulty}_{solver_config.tag}.png",
)
