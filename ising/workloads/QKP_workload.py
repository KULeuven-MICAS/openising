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
top_benchmark = "./ising/benchmarks/Knapsack/"
solver_config = SolverConfig()
config_file = "./ising/inputs/config/config_qkpWorkload.yaml"
difficulty = "easy"  # easy - 100 items, medium - 200 items, difficult - 300 items

# Each bucket spans 4 densities (25/50/75/100 where available) and varies the instance id
# so the 6 problems cover different structures within the same size class.
if difficulty == "easy":
    problems = [
        "jeu_100_25_1.txt", "jeu_100_50_1.txt", "jeu_100_75_1.txt",
        "jeu_100_100_1.txt", "jeu_100_25_3.txt", "jeu_100_75_3.txt",
    ]
elif difficulty == "medium":
    problems = [
        "jeu_200_25_1.txt", "jeu_200_50_1.txt", "jeu_200_75_1.txt",
        "jeu_200_100_1.txt", "jeu_200_25_3.txt", "jeu_200_75_3.txt",
    ]
elif difficulty == "difficult":
    # Knapsack/jeu_300_* only has densities 25 and 50, so vary instance ids instead.
    problems = [
        "jeu_300_25_1.txt", "jeu_300_25_2.txt", "jeu_300_25_3.txt",
        "jeu_300_50_1.txt", "jeu_300_50_2.txt", "jeu_300_50_3.txt",
    ]
else:
    raise ValueError("Invalid difficulty level. Options are 'easy', 'medium', or 'difficult'.")

ans_list = []
for problem in problems:
    with (TOP / config_file).open("r") as f:
        config = yaml.safe_load(f)
    config["benchmark"] = top_benchmark + problem
    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)
    ans = run_workload(problem_type="QKP", solver_config=solver_config, config_file=config_file)
    ans_list.append(ans)
summarize_workload(
    output_file=TOP / f"ising/workloads/qkp_results_{difficulty}_{solver_config.tag}.out",
    problem_type="QKP",
    config_path=config_file,
    ans_list=ans_list,
)

df = ans_to_metric_df({"all": ans_list}, label_name="bucket", problem="QKP")
box_plot_metric(
    df,
    x="benchmark",
    problem="QKP",
    title=f"QKP - {difficulty} difficulty, {solver_config.tag} solver",
    save_path=TOP / f"ising/workloads/qkp_boxplot_{difficulty}_{solver_config.tag}.png",
)
