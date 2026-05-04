# Workload generator

Run the Galena (Multiplicative) solver across a curated set of benchmarks for any of
four COP problem types and any of five solver settings, and produce a comparison plot
of solution quality.

## What you can test

| Axis            | Options                                                                |
|-----------------|------------------------------------------------------------------------|
| Problem type    | `Maxcut`, `TSP`, `QKP`, `Biqmac`, `MIMO`                               |
| Difficulty      | `easy`, `medium` (TSP/QKP/Biqmac/MIMO), `difficult`                    |
| Solver setting  | `base`, `hw`, `hw_cn`, `hw_mc`, `hw_cn_mc` (see SolverConfig below)    |

Each per-problem workload runs the solver across 6 benchmarks (varied in size,
density, and structure within a difficulty bucket) for `nb_runs` trials each
(`nb_trials` for MIMO). The solver-quality metric is gap to best known
`|E - E*| / |E*|`, except for MIMO where it is per-trial BER. MIMO uses the dummy
creator instead of benchmark files: difficulty buckets vary `(user_num, ant_num,
QAM)` while keeping the Ising variable count at or under 256.

## Quick start

The typical flow is **edit one workload script → run it once per setting → run
`plot_workload.py`**. No solver code needs to change.

1. Open the workload script for the problem you want, e.g. [Maxcut_workload.py](Maxcut_workload.py).
2. Edit the three knobs at the top:
   - `solver_config` — which Galena features to enable (see below)
   - `difficulty` — `"easy"`, `"medium"`, or `"difficult"`
   - `config_file` — path to the YAML config (already wired)
3. Run it: `python -m ising.workloads.Maxcut_workload`. Repeat for each
   `solver_config` you want to compare.
4. Run [plot_workload.py](plot_workload.py) (set `problem_type` at the bottom) to
   load every saved `.ans` file for that problem type and produce a single
   box-plot comparison: x-axis = benchmark, hue = solver setting, log-y gap.

## SolverConfig

Defined in [run_workload.py](run_workload.py). Three independent booleans control
the five legal solver settings:

```python
SolverConfig()                                              # base, no HW assumptions
SolverConfig(hw=True)                                       # HW assumptions on
SolverConfig(hw=True, comb_nodes=True)                      # HW + combine-nodes
SolverConfig(hw=True, multi_core=True)                      # HW + multi-core
SolverConfig(hw=True, comb_nodes=True, multi_core=True)     # both improvements
```

- **hw**: turns on quantization, mismatch, and delay (toggle sub-flags in the YAML).
- **comb_nodes**: represents one node by a combination of nodes to raise quantization precision.
- **multi_core**: splits the model across cores to scale node count. *Currently not implemented in the solver.*

`comb_nodes` and `multi_core` require `hw=True` (raises `ValueError` otherwise).

Each `SolverConfig` has a `.tag` property (`base`, `hw`, `hw_cn`, `hw_mc`,
`hw_cn_mc`) used for filenames and plot labels.

## WorkloadSettings

Tunables written into the YAML before each run live in `WorkloadSettings`
(also in [run_workload.py](run_workload.py)). Default values are sensible —
override only what you need:

```python
ans, _ = run_workload(
    problem_type="Maxcut",
    solver_config=SolverConfig(hw=True),
    config_file=config_file,
    settings=WorkloadSettings(mismatch_std=0.2, quantization_precision=6),
)
```

Mistyped field names raise immediately at construction.

## Outputs

- Comparison figure from `plot_workload.py`: `ising/outputs/<problem>/figures/<problem>_galena_comparison.png`
- `.ans` files persist across runs — re-running with a different `solver_config` adds to the comparison rather than overwriting it.
