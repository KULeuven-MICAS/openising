"""
Aggregate every saved Ans file under ising/outputs/<problem_type>/ans and produce a
comparison box plot across solver settings.

Run this after the per-solver workload scripts (Maxcut_workload.py, TSP_workload.py,
QKP_workload.py, Biqmac_workload.py) have been executed for each SolverConfig you want
to compare. The resulting figure groups by benchmark on the x-axis and uses hue for
each Galena setting, so the impact of HW assumptions and the comb_nodes / multi_core
improvements is directly visible per benchmark.
"""
from ising.stages import TOP, LOGGER
from ising.stages.simulation_stage import Ans
from ising.postprocessing.summarize_energies import ans_to_metric_df, box_plot_metric
from ising.workloads.run_workload import SolverConfig


def _tag_from_stem(stem: str) -> str | None:
    """Return the SolverConfig tag suffix of `stem` (after the last `_`), or None.

    Saved filenames are `<benchmark>_<tag>.ans`. Benchmark names may contain
    underscores; tags are short tokens drawn from {hw, cn, mc, base} so the tag
    is always the longest valid suffix that parses via SolverConfig.from_tag.
    """
    parts = stem.split("_")
    for split in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[split:])
        try:
            SolverConfig.from_tag(candidate)
        except ValueError:
            continue
        return candidate
    return None


def load_workload_ans(problem_type: str) -> dict[str, list[Ans]]:
    """Load every .ans file under ising/outputs/<problem_type>/ans, grouped by tag."""
    folder = TOP / f"ising/outputs/{problem_type}/ans"
    if not folder.exists():
        raise FileNotFoundError(f"No saved Ans files at {folder}. Run the workload first.")

    grouped: dict[str, list[Ans]] = {}
    for path in sorted(folder.glob("*.ans")):
        tag = _tag_from_stem(path.stem)
        if tag is None:
            LOGGER.warning(f"Skipping {path.name}: filename does not end in a known SolverConfig tag.")
            continue
        ans = Ans()
        ans.load(path)
        grouped.setdefault(tag, []).append(ans)
    return grouped


if __name__ == "__main__":
    problem_type = "Maxcut"  # one of: "Maxcut", "TSP", "QKP", "Biqmac", "MIMO"

    grouped = load_workload_ans(problem_type)
    if not grouped:
        raise RuntimeError(f"No Ans files matched a known SolverConfig tag for {problem_type}.")

    df = ans_to_metric_df(grouped, label_name="solver_type", problem=problem_type)

    save_dir = TOP / f"ising/outputs/{problem_type}/figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{problem_type}_galena_comparison.png"
    box_plot_metric(
        df,
        x="benchmark",
        problem=problem_type,
        hue="solver_type",
        title=f"{problem_type} - solver setting comparison",
        save_path=save_path,
    )
    LOGGER.info(f"Saved comparison plot to {save_path}")
