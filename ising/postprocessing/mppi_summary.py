from pathlib import Path
from ising.stages.model.MPPI.environment import create_environment
from .plot_mppi_trajectory import plot_results


def summarize_mppi(output_dir: Path, ans, name: str | None = None):
    env, _, _ = create_environment(ans.scene)
    x_ref = ans.reference_trajectory

    Path.mkdir(output_dir, parents=True, exist_ok=True)
    if name is None:
        plot_results(
            env, x_ref, ans.executed_trajectory, ans.predicted_trajectory, savefile=output_dir / "mppi_results.png"
        )
    else:
        plot_results(
            env, x_ref, ans.executed_trajectory, ans.predicted_trajectory, savefile=output_dir / f"mppi_{name}.png"
        )
