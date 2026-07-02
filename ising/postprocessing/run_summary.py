import logging
from pathlib import Path
import numpy as np

from ising.utils.flow import compute_ttt, approximation_to_best_found
from ising.utils.problem_difficulty import compute_ruggedness
from ising.stages.simulation_stage import Ans

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
        mean_acc = " ".join([f"{np.mean(accuracies[solver])}" for solver in ans_list[0].config.solvers])
        mean_tts = " ".join([f"{np.mean(tts_all[solver])}" for solver in ans_list[0].config.solvers])

        with Path.open(output_file, "a") as f:
            f.write("=====Summary of all runs=====\n")
            f.write(f"mean approximation value| {mean_acc}\n")
            f.write(f"mean TTT 0.9| {mean_tts}\n")
    else:
        mean_ber = " ".join([f"{np.mean(bers[solver])}" for solver in ans_list[0].config.solvers])
        with Path.open(output_file, "a") as f:
            f.write("=====Summary of all runs=====\n")
            f.write(f"mean BER| {mean_ber}\n")

def summarize_runs(output_file: Path, ans: Ans, problem_type: str, config_path: str):
    solvers = ans.config.solvers
    mean_computation_time = {solver: np.mean(ans.computation_time[solver]) for solver in solvers}
    comp_str = " ".join([f"{mean_computation_time[solver]:.4e}s" for solver in solvers])
    solver_str = " ".join(solvers)
    operation_str = " ".join([f"{ans.operation_count[solver]}" for solver in solvers])
    operation_it_str = " ".join(
        [
            f"{
                ans.operation_count[solver]
                / ans.config.__getattribute__(
                    f'num_iterations_{solver}' if solver not in ['bSB', 'dSB'] else 'num_iterations_SB'
                )
            }"
            for solver in solvers
        ]
    )
    ruggednes_prob = compute_ruggedness(ans.ising_model, ans.ising_model.num_variables*10)
    if problem_type == "MIMO":
        logging.info("BER: %s", ans.BER)
        if not ans.config.dummy_creator:
            BER_str = " ".join([str(ans.BER[solver]) for solver in solvers])
            with Path.open(output_file, "a") as f:
                f.write("\n")
                f.write("=====================\n")
                f.write(f"results of running {ans.benchmark} with {config_path.rsplit('/', maxsplit=1)[-1]}:\n")
                f.write(f"logfile discriminator: {ans.config.logfile_discrimination}\n")
                f.write("=====================\n")
                f.write(f"ruggedness {ans.benchmark}| {ruggednes_prob}\n")
                f.write("=====================\n")
                f.write("MIMO results:\n")
                f.write(f"SNR|BER  {solver_str}\n")
                f.write(f"{ans.SNR}|     {BER_str}\n")
                f.write("=====================\n")
                f.write("Simulation results:\n")
                f.write(f"solver| {solver_str}\n")
                f.write(f"computation time| {comp_str}\n")
                f.write(f"operation count| {operation_str}\n")
                f.write(f"operation count / it| {operation_it_str}\n")
        return None
    elif not ans.config.dummy_creator:
        benchmark = ans.benchmark
        ising_energies = ans.energies
        best_found = ans.best_found
        ising_energy_max = {solver: np.max(ising_energies[solver]) for solver in solvers}
        max_en_str = " ".join([f"{ising_energy_max[solver]:.4f}" for solver in solvers])
        ising_energy_min = {solver: np.min(ising_energies[solver]) for solver in solvers}
        min_en_str = " ".join([f"{ising_energy_min[solver]:.4f}" for solver in solvers])
        ising_energy_avg = {solver: np.mean(ising_energies[solver]) for solver in solvers}
        avg_en_str = " ".join([f"{ising_energy_avg[solver]:.4f}" for solver in solvers])

        tts = {
            solver: compute_ttt(ising_energies[solver], mean_computation_time[solver], best_found, ans.config.nb_runs)
            for solver in solvers
        }
        approximation = {
            solver: approximation_to_best_found(np.array(ising_energy_avg[solver]), best_found) for solver in solvers
        }
        approx_str = " ".join([f"{approximation[solver]:.2f}%" for solver in solvers])
        tts_str = " ".join([f"{tts[solver]:.4e}s" for solver in solvers])
        with Path.open(output_file, "a") as f:
            f.write("\n")
            f.write("=====================\n")
            f.write(f"results of running {benchmark} with {config_path.rsplit('/', maxsplit=1)[-1]}:\n")
            f.write(f"logfile discriminator: {ans.config.logfile_discrimination}\n")
            f.write(f"reference energy {best_found}\n")
            f.write("=====================\n")
            f.write(f"ruggedness {benchmark}| {ruggednes_prob}\n")
            f.write("=====================\n")
            f.write("Simulation results:\n")
            f.write(f"solver| {solver_str}\n")
            f.write(f"energy max| {max_en_str}\n")
            f.write(f"energy min| {min_en_str}\n")
            f.write(f"energy avg| {avg_en_str}\n")
            f.write(f"computation time| {comp_str}\n")
            f.write(f"TTT 0.9| {tts_str}\n")
            f.write(f"operation count| {operation_str}\n")
            f.write(f"operation count / it| {operation_it_str}\n")
            f.write(f"Approximation value| {approx_str}\n")
            f.write("\n")

        logging.info(
            "benchmark: %s, \n ruggedness: %s, \n reference: %s,\n energy max: %s, \n min: %s, \n avg: %s",
            benchmark,
            ruggednes_prob,
            best_found,
            ising_energy_max,
            ising_energy_min,
            ising_energy_avg,
        )
        return tts, approximation
