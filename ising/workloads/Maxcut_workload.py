from ising.workloads.run_workload import run_workload, SolverConfig
from ising.stages import TOP
from ising.postprocessing.run_summary import summarize_workload
from ising.postprocessing.summarize_energies import ans_to_metric_df, box_plot_metric
import yaml
from ising.utils.problem_difficulty import compute_ruggedness

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
top_benchmark = "./ising/benchmarks/G/"
solver_config = SolverConfig()
config_file = "./ising/inputs/config/config_mcWorkload.yaml"
difficulty = "easy"  # easy - 800 nodes, difficult - 2000 nodes

if difficulty == "easy":
    problems = ["G1.txt", "G6.txt", "G11.txt", "G14.txt", "G18.txt"]
elif difficulty == "difficult":
    problems = ["K2000.txt", "G22.txt", "G27.txt", "G32.txt", "G35.txt", "G39.txt"]
else:
    raise ValueError("Invalid difficulty level. Options are 'easy' or 'difficult'.")

ans_list = []
for problem in problems:
    with (TOP / config_file).open("r") as f:
        config = yaml.safe_load(f)
    config["benchmark"] = top_benchmark + problem
    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)
    ans = run_workload(problem_type="Maxcut", solver_config=solver_config, config_file=config_file)
    compute_ruggedness(ans.ising_model, 10000)
    ans_list.append(ans)
summarize_workload(
    output_file=TOP / f"ising/workloads/maxcut_results_{difficulty}_{solver_config.tag}.out",
    problem_type="Max Cut",
    config_path=config_file,
    ans_list=ans_list,
)

df = ans_to_metric_df({"all": ans_list}, label_name="bucket", problem="Maxcut")
box_plot_metric(
    df,
    x="benchmark",
    problem="Maxcut",
    title=f"Max Cut - {difficulty} difficulty, {solver_config.tag} solver",
    save_path=TOP / f"ising/workloads/maxcut_boxplot_{difficulty}_{solver_config.tag}.png",
)
