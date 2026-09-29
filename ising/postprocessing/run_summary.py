import logging
from pathlib import Path
import numpy as np

from ising.utils.flow import compute_ttt, relative_to_best_found, approximation_to_best_found
from ising.utils.problem_difficulty import compute_ruggedness
from ising.stages.simulation_stage import Ans
from ising.postprocessing.helper_functions import get_string
from ising.postprocessing.plot_model import plot_model, plot_model_distribution
from ising.stages.model.MPPI.environment import create_environment

from ising.postprocessing.plot_mppi_trajectory import plot_results


def summarize_workload(output_file: Path, problem_type: str, config_path: str, ans_list: list[Ans]):
    accuracies = {solver: [] for solver in ans_list[0].config.solvers}
    tts_all = {solver: [] for solver in ans_list[0].config.solvers}
    bers = {solver: [] for solver in ans_list[0].config.solvers}
    for ans in ans_list:
        ret = summarize_runs(output_file, ans, problem_type, config_path)
        if problem_type != "MIMO":
            for solver in ans.config.solvers:
                accuracies[solver].append(ret[1][solver])
                tts_all[solver].append(ret[0][solver])
        else:
            for solver in ans.config.solvers:
                bers[solver].append(ans.BER)
    if problem_type != "MIMO":
        mean_acc = get_string(accuracies, ans_list[0].config.solvers, np.mean)
        mean_tts = get_string(tts_all, ans_list[0].config.solvers, np.mean)

        with Path.open(output_file, "a") as f:
            f.write("=====Summary of all runs=====\n")
            f.write(f"mean approximation value| {mean_acc}\n")
            f.write(f"mean TTT 0.9| {mean_tts}\n")

    else:
        mean_ber = get_string(bers, ans_list[0].config.solvers, np.mean)
        with Path.open(output_file, "a") as f:
            f.write("=====Summary of all runs=====\n")
            f.write(f"mean BER| {mean_ber}\n")

def add_metric(solvers, label, values, reduce=lambda x: x):
    return f"{label}| {get_string(values, solvers, reduce)}"

def summarize_runs(output_file: Path, ans: Ans, problem_type: str, config_path: str):
    solvers = ans.config.solvers
    solver_str = " ".join(solvers)
    separator = "====================="
    lines = [
        "", separator,
        f"results of running {ans.benchmark} with {config_path.rsplit('/', maxsplit=1)[-1]}:",
        f"logfile discriminator: {ans.config.logfile_discrimination}",
        separator,
    ]

    ret = None
    metrics = {}
    if problem_type == "MIMO":
        logging.info("BER: %s", ans.BER)
        lines.extend([
            "MIMO results:", f"SNR|BER  {solver_str}",
            f"{ans.SNR}|     {get_string(ans.BER, solvers, lambda x: x)}",
        ])
    elif problem_type == 'MPPI':
        env, _, _ = create_environment(ans.scene)
        x_ref = ans.reference_trajectory
        output_dir = output_file.parent
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        plot_results(
            env, x_ref, ans.executed_trajectory, ans.predicted_trajectory, savefile=output_dir / "mppi_results.png"
        )
        res = ans.executed_trajectory - ans.reference_trajectory
        rmse = np.sqrt(res ** 2).mean()
        r2 = 1 - (res ** 2 / np.maximum((res ** 2).mean(), 10e-4)).mean()

        lines.append("=====Accuracy of solution =====\n")
        lines.append(f"rMSE: {rmse:.4f} \n")
        lines.append(f"r-squared: {r2:.4f} \n")
    else:
        energy_stats = {
            label: {solver: reduce(ans.energies[solver]) for solver in solvers}
            for label, reduce in (("max", np.max), ("min", np.min), ("avg", np.mean))
        }
        ruggedness = compute_ruggedness(ans.ising_model, ans.ising_model.num_variables * 500)
        lines.append(f"ruggedness {ans.benchmark}| {ruggedness}")
        best_found = None if ans.config.dummy_creator else ans.best_found
        if ans.config.dummy_creator:
            lines.append(f"Dummy problem of size {ans.ising_model.num_variables}")
        else:
            lines.append(f"reference energy {best_found}")
            tts = {
                solver: compute_ttt(
                    ans.energies[solver], np.mean(ans.computation_time[solver]), best_found, ans.config.nb_runs
                )
                for solver in solvers
            }
            relative = {
                solver: relative_to_best_found(value, best_found) for solver, value in energy_stats["avg"].items()
            }
            approximation = {
                solver: approximation_to_best_found(np.array(value), best_found)
                for solver, value in energy_stats["avg"].items()
            }
            metrics = {"TTT 0.9": tts, "Relative error": relative, "Approximation value": approximation}
            ret = tts, approximation
        logging.info(
            "benchmark: %s, \n ruggedness: %s, \n reference: %s,\n energy max: %s, \n min: %s, \n avg: %s",
            ans.benchmark, ruggedness, best_found,
            energy_stats["max"], energy_stats["min"], energy_stats["avg"],
        )
        plot_model(ans.ising_model, f"{ans.benchmark}_model")
        plot_model_distribution(ans.ising_model, f"{ans.benchmark}_model_distribution")
        if ans.config.quantization:
            plot_model(ans.quantized_model, f"{ans.benchmark}_quantized_model")
            plot_model_distribution(ans.quantized_model, f"{ans.benchmark}_quantized_model_distribution")

    lines.extend([separator, "Simulation results:", f"solver| {solver_str}"])
    for label, values in metrics.items():
        lines.append(add_metric(solvers, label, values))
    if problem_type != "MIMO":
        for label, values in energy_stats.items():
            lines.append(add_metric(solvers, f"energy {label}", values))
        lines.append("")
    lines.append(add_metric(solvers, "computation time", ans.computation_time, np.mean))
    lines.append(add_metric(solvers, "operation count", ans.operation_count))
    lines.append(add_metric(solvers, "operation count / it", {
        solver: ans.operation_count[solver] / getattr(
            ans.config, f"num_iterations_{'SB' if solver in ('bSB', 'dSB') else solver}"
        )
        for solver in solvers
    }))
    with Path.open(output_file, "a") as f:
        f.write("\n".join(lines) + "\n")
    return ret
