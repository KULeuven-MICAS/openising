from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from ising.api import get_hamiltonian_energy
from ising.stages import TOP, LOGGER


@dataclass(frozen=True)
class WorkloadSettings:
    """Tunables that run_workload writes into the YAML config before launching the API.

    Workload scripts override only the fields they care about; everything else uses the
    defaults below. Frozen so a shared default instance is safe to reuse.
    """
    current: float = 1e-6
    capacitance: float = 1e-15
    quantization: bool = True
    quantization_precision: int = 4
    mismatch_std: float = 0.1
    sigma_J: float = 0.0
    accumulation_delay: float = 0.0
    broadcast_delay: float = 0.0
    delay_offset: float = 0.0
    nodes_scaling: int = 2
    nb_cores: int = 2


@dataclass(frozen=True)
class SolverConfig:
    """Which Galena solver features are enabled.

    The five legal combinations are:
      - SolverConfig() — base solver, no HW assumptions
      - SolverConfig(hw=True) — base solver under HW assumptions
      - SolverConfig(hw=True, comb_nodes=True) — HW + combine-nodes improvement
      - SolverConfig(hw=True, multi_core=True) — HW + multi-core improvement
      - SolverConfig(hw=True, comb_nodes=True, multi_core=True) — both improvements

    `comb_nodes` and `multi_core` require `hw=True`; either alone raises.
    """
    hw: bool = False
    comb_nodes: bool = False
    multi_core: bool = False

    def __post_init__(self):
        if (self.comb_nodes or self.multi_core) and not self.hw:
            raise ValueError("comb_nodes and multi_core require hw=True")

    @property
    def tag(self) -> str:
        """Short token for filenames / plot labels. 'base' or 'hw[_cn][_mc]'."""
        if not self.hw:
            return "base"
        parts = ["hw"]
        if self.comb_nodes:
            parts.append("cn")
        if self.multi_core:
            parts.append("mc")
        return "_".join(parts)

    @classmethod
    def from_tag(cls, tag: str) -> "SolverConfig":
        """Inverse of `tag`. Raises ValueError on unknown tokens."""
        if tag == "base":
            return cls()
        tokens = set(tag.split("_"))
        unknown = tokens - {"hw", "cn", "mc"}
        if unknown or "hw" not in tokens:
            raise ValueError(f"Unknown SolverConfig tag {tag!r}")
        return cls(hw=True, comb_nodes="cn" in tokens, multi_core="mc" in tokens)


_BASE_FIELDS = ("current", "capacitance")
_HW_FIELDS = (
    "quantization", "quantization_precision", "mismatch_std", "sigma_J",
    "accumulation_delay", "broadcast_delay", "delay_offset",
)
_COMB_NODES_FIELDS = ("nodes_scaling",)
_MULTI_CORE_FIELDS = ("nb_cores",)


def run_workload(problem_type, solver_config: SolverConfig, config_file,
                 settings: WorkloadSettings = WorkloadSettings(),
                 benchmark_label: str | None = None):
    """
    Runs the specified workload with the given solver configuration and YAML file.

    @type problem_type: str
    @param problem_type: The type of problem to solve (e.g., "Maxcut", "TSP", etc.).
    @type solver_config: SolverConfig
    @param solver_config: Which Galena solver features to enable.
    @type config_file: str
    @param config_file: The path to the configuration file for the workload.
    @type settings: WorkloadSettings
    @param settings: Tunables to write into the config. Defaults are used
        for any field the caller does not override.
    @type benchmark_label: str | None
    @param benchmark_label: optional override for C{ans.benchmark}. Useful for the MIMO
        workload, where the dummy creator yields a generic "dummy_MIMO" label that would
        otherwise collide across iterations. When provided, it is also used as the save
        filename stem.
    @return: A tuple C{(ans, debug_info)} — the answer obtained from running the workload
        and debug information collected during the run.
    """
    values = asdict(settings)
    fields_to_apply = list(_BASE_FIELDS)
    extra = {}
    if solver_config.hw:
        fields_to_apply.extend(_HW_FIELDS)
    if solver_config.comb_nodes:
        extra["combine_nodes"] = True
        fields_to_apply.extend(_COMB_NODES_FIELDS)
    if solver_config.multi_core:
        extra["multi_core"] = True
        fields_to_apply.extend(_MULTI_CORE_FIELDS)

    config_updates = {f: values[f] for f in fields_to_apply}
    config_updates.update(extra)

    with (TOP / config_file).open() as f:
        config = yaml.safe_load(f)
    config.update(config_updates)

    if config.get("solvers") != ["Multiplicative"]:
        LOGGER.warning(
            f"Workload runs only support the Multiplicative solver, but config has solvers={config.get('solvers')}. "
            "Overriding to ['Multiplicative']."
        )
        config["solvers"] = ["Multiplicative"]

    with (TOP / config_file).open("w") as f:
        yaml.safe_dump(config, f)

    ans, debug_info = get_hamiltonian_energy(problem_type=problem_type, config_path=config_file)
    if benchmark_label is not None:
        ans.benchmark = benchmark_label
    output_folder = TOP / f"ising/outputs/{problem_type}/ans"
    if not output_folder.exists():
        Path.mkdir(output_folder)
    ans.save(output_folder / f"{ans.benchmark}_{solver_config.tag}.ans")
    return ans, debug_info
