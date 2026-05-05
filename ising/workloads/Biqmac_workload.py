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
top_benchmark = "./ising/benchmarks/biqmac/"
solver_config = SolverConfig()
config_file = "./ising/inputs/config/config_biqmacWorkload.yaml"
difficulty = "easy"  # easy - ~50-100 nodes, medium - ~150-250 nodes, difficult - >=500 nodes

# Within each bucket the 6 problems are pulled from different sub-corpora (be / beasley /
# gka / rudy) so weight distributions and graph density vary even when the node count is
# similar. Rudy variants in particular cover g05 (Erdős-Rényi p=0.5), pm1s/pm1d (sparse
# / dense ±1 weights), and pw01/w05 (weighted) structures.
if difficulty == "easy":
    problems = [
        "beasley/bqp50-1.sparse",
        "beasley/bqp100-1.sparse",
        "be/be100.1.sparse",
        "rudy/g05_100.0",
        "rudy/pm1s_100.0",
        "rudy/pw01_100.0",
    ]
elif difficulty == "medium":
    problems = [
        "be/be120.3.1.sparse",
        "be/be150.3.1.sparse",
        "be/be150.8.1.sparse",
        "be/be200.3.1.sparse",
        "be/be250.1.sparse",
        "beasley/bqp250-1.sparse",
    ]
elif difficulty == "difficult":
    problems = [
        "beasley/bqp500-1.sparse",
        "beasley/bqp500-5.sparse",
        "beasley/bqp1000-2.sparse",
        "beasley/bqp1000-5.sparse",
        "beasley/bqp2500-1.sparse",
        "beasley/bqp2500-5.sparse",
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
    ans = run_workload(problem_type="Biqmac", solver_config=solver_config, config_file=config_file)
    ans_list.append(ans)
summarize_workload(
    output_file=TOP / f"ising/workloads/biqmac_results_{difficulty}_{solver_config.tag}.out",
    problem_type="Biqmac",
    config_path=config_file,
    ans_list=ans_list,
)

df = ans_to_metric_df({"all": ans_list}, label_name="bucket", problem="Biqmac")
box_plot_metric(
    df,
    x="benchmark",
    problem="Biqmac",
    title=f"Biqmac - {difficulty} difficulty, {solver_config.tag} solver",
    save_path=TOP / f"ising/workloads/biqmac_boxplot_{difficulty}_{solver_config.tag}.png",
)
