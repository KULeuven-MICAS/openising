from ising.workloads.run_workload import workload_api, SolverConfig
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

workload_api(
    problem_type="Maxcut",
    problem_label="Max Cut",
    solver_config=solver_config,
    config_file=config_file,
    difficulty=difficulty,
    benchmarks=[({"benchmark": top_benchmark + p}, None) for p in problems],
    on_ans=lambda ans: compute_ruggedness(ans.ising_model, 10000),
    simulation=False,
)
