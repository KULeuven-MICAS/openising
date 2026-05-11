from ising.workloads.run_workload import workload_api, SolverConfig

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

The dummy creator is always enabled for this workload, so MIMO instances are
synthesised at run time from the (user_num, ant_num, QAM) tuple instead of being
read from a benchmark file. The Ising variable count is `r * 2 * user_num` where
`r = log2(sqrt(M))`; every instance below stays at or under 256 variables.
"""
solver_config = SolverConfig()
config_file = "./ising/inputs/config/config_mimoWorkload.yaml"
difficulty = "easy"  # easy - QPSK with antennas >= users; medium - 16-QAM; difficult - 64/256-QAM at tight ratios

# Each entry is (user_num, ant_num, M). Entries vary the user-to-antenna ratio and the
# QAM scheme inside each bucket so multiple difficulty drivers are exercised.
if difficulty == "easy":
    # M=4 (QPSK), vars = 2 * user_num. Antennas >= users (favorable detection).
    instances = [
        (4, 8, 4), (8, 16, 4), (16, 32, 4),
        (32, 64, 4), (64, 128, 4), (128, 128, 4),
    ]
elif difficulty == "medium":
    # M=16, vars = 4 * user_num. Mix of favorable and tight user/antenna ratios.
    instances = [
        (4, 8, 16), (8, 16, 16), (16, 32, 16),
        (32, 64, 16), (32, 32, 16), (64, 64, 16),
    ]
elif difficulty == "difficult":
    # M=64 and M=256 with tight or 1:1 user/antenna ratios.
    # Vars = 6 * user_num (M=64) or 8 * user_num (M=256).
    instances = [
        (8, 16, 64), (16, 32, 64), (32, 64, 64),
        (32, 32, 64), (16, 16, 256), (32, 32, 256),
    ]
else:
    raise ValueError("Invalid difficulty level. Options are 'easy', 'medium', or 'difficult'.")

workload_api(
    problem_type="MIMO",
    problem_label="MIMO",
    solver_config=solver_config,
    config_file=config_file,
    difficulty=difficulty,
    benchmarks=[
        (
            {
                "dummy_creator": True,
                "dummy_user_num": u,
                "dummy_ant_num": a,
                "dummy_qam": M,
                "nb_runs": 1,
            },
            f"u{u}_a{a}_M{M}",
        )
        for u, a, M in instances
    ],
)
