import logging
import sys
import os
import yaml

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import numpy as np
from ising.stages import TOP
from ising import api
from scipy.stats import gmean
from ising.stages.simulation_stage import Ans
from ising.postprocessing.run_summary import summarize_runs
from ising.utils.flow import relative_to_best_found

import matplotlib.pyplot as plt
from Paper_OpenIsing import BASE_PATH


def change_yticks(ax, problem_type: str, best_found: int, large_graph=True):
    if large_graph:
        yticks = ax.get_yticks()[1:-1]
    else:
        yticks = ax.get_yticks()[:-1]
    diff = np.abs(yticks - best_found)
    idx_closest = np.argmin(diff)
    yticks = np.sort(np.append(np.delete(yticks, idx_closest), best_found))
    ax.set_yticks(yticks)
    if problem_type != "Maxcut":
        lab_ax = [f"{t:.1e}" if t != best_found else str(int(t)) for t in yticks]
    else:
        lab_ax = [f"{t:.2e}" if t != best_found else str(int(t)) for t in yticks]
    ax.set_yticklabels(lab_ax, fontsize=18 if large_graph else 15)
    ax.tick_params(axis="both", which="major", labelsize=18 if large_graph else 15)
    for tl in ax.get_yticklabels():
        if tl._y == best_found:
            tl.set_color("red")


# Initialize the logger
logging_level = logging.WARNING
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_types = ["Maxcut", "Biqmac", "TSP", "QKP"]  # Specify the problem type [Maxcut, TSP, ATSP, MIMO] "MIMO",
# Initialize all paths
base_path = BASE_PATH / "Pareto_curve_OPS"
data_path = base_path / "data"
data_path.mkdir(parents=True, exist_ok=True)
config_path = base_path / "config"
fig_path = base_path / "figures"

with (config_path / "pareto_curves_Maxcut.yaml").open("r") as f:
    config = yaml.safe_load(f)
nb_benchmarks = len(config["benchmark"])
nb_runs = config["nb_runs"]
nb_points = len(config["num_iterations_SA"])
solvers = list(config["solvers"])

ans_all = {
    problem: {point: [Ans() for _ in range(nb_benchmarks)] for point in range(nb_points)} for problem in problem_types
}
for problem_type in problem_types:
    config_path_problem = config_path / f"pareto_curves_{problem_type}.yaml"
    with config_path_problem.open("r") as f:
        new_config = yaml.safe_load(f)
    num_it_solvers = {
        solver: list(
            new_config[f"num_iterations_{solver}" if (solver != "bSB" and solver != "dSB") else "num_iterations_SB"]
        )
        for solver in solvers
    }
    benchmarks = new_config["benchmark"]
    for ind, benchmark in enumerate(benchmarks):
        new_config["benchmark"] = benchmark
        for point in range(nb_points):
            data_file = data_path / f"{problem_type}_benchmark{ind}_point{point}.pkl"
            if data_file.exists():
                ans_all[problem_type][point][ind].load(data_file)
            else:
                for solver in solvers:
                    new_config[
                        f"num_iterations_{solver}" if (solver != "bSB" and solver != "dSB") else "num_iterations_SB"
                    ] = num_it_solvers[solver][point]
                file = config_path / f"config_pareto_curve_{problem_type}_point{point}.yaml"
                with file.open("w") as f:
                    yaml.dump(new_config, f)
                ans, debug_info = api.get_hamiltonian_energy(
                    problem_type=problem_type,
                    config_path=str(file.relative_to(TOP)),
                    logging_level=logging_level,
                )
                ans_all[problem_type][point][ind] = ans
                ans.save(data_file)
markers = ["s", "d", "^", "o", "p", "v"]

fig_op, axes = plt.subplots(1, len(problem_types), figsize=(28, 5))
labels = dict()
for problem_type, ax in zip(problem_types, axes):
    output_file = base_path / f"simulation_summary_{problem_type}.txt"
    mean_en = {solver: [] for solver in solvers}
    std_en = {solver: [] for solver in solvers}
    ops = {solver: [] for solver in solvers}
    for point in range(nb_points):
        with output_file.open("a") as f:
            f.write("=================================================\n")
            f.write(f"======= Simulation point {point} =======\n")
        ans_list = ans_all[problem_type][point]
        energies = {solver: np.array([]) for solver in solvers}
        operations = {solver: [] for solver in solvers}
        for ans_bench in ans_list:
            summarize_runs(
                output_file,
                ans_bench,
                problem_type,
                str((config_path / f"config_pareto_curve_{problem_type}_point{point}.yaml").relative_to(TOP)),
            )

            for solver in solvers:
                energies[solver] = np.append(
                    energies[solver], relative_to_best_found(np.array(ans_bench.energies[solver]), ans_bench.best_found)
                )
                operations[solver].append(ans_bench.operation_count[solver])
        for solver in solvers:
            mean = np.mean(energies[solver])
            mean_en[solver].append(mean)
            std_en[solver].append(np.std(energies[solver])/mean)
            ops[solver].append(gmean(operations[solver]))

    for solver in solvers:
        lab = solver + int(solver == "BRIM") * " (estimated)"
        err = ax.plot(
            ops[solver],
            mean_en[solver],
            linestyle="-" if solver != "BRIM" else "--",
            marker=markers[solvers.index(solver)],
            alpha=1 if solver != "BRIM" else 0.7,
            label=lab,
            markersize=10,
        )
        ax.fill_between(ops[solver],
                        np.array(mean_en[solver]) - np.array(std_en[solver]),
                        np.array(mean_en[solver]) + np.array(std_en[solver]),
                        alpha=0.3)
        if len(err) >= 1:
            labels[lab] = err[0]
        else:
            labels[lab] = err

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Operation Count", fontsize=20)
    if problem_type == "Maxcut":
        problem_type_ = "Max Cut"
        ax.set_ylabel("Relative Ising Energy to Best Found", fontsize=20)
    else:
        problem_type_ = problem_type
    ax.set_title(problem_type_)
fig_op.tight_layout()
fig_op.legend(
    list(labels.values()),
    list(labels.keys()),
    fontsize=20,
    loc="upper center",
    ncol=len(solvers) + 2,
    fancybox=True,
    bbox_to_anchor=(0.5, 1.1),
    columnspacing=0.3,
)
fig_op.savefig(fig_path / "pareto_curve.pdf", bbox_inches="tight")
plt.close()
