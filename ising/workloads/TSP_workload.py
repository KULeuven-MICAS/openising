from ising.workloads.run_workload import workload_api, SolverConfig, WorkloadSettings

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
solver_config = SolverConfig(hw=True)
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

settings = WorkloadSettings(mismatch_std=0.0, nodes_scaling=1, nb_cores=1)

workload_api(
    problem_type="TSP",
    problem_label="TSP",
    solver_config=solver_config,
    config_file=config_file,
    difficulty=difficulty,
    settings=settings,
    simulation=True,
    benchmarks=[({"benchmark": top_benchmark + p}, None) for p in problems],
)
