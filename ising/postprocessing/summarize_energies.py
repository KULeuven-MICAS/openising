import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from scipy.stats import gmean


from ising.utils.HDF5Logger import return_metadata
from ising.postprocessing.helper_functions import get_metadata_from_logfiles
from ising.stages.simulation_stage import Ans
from ising.utils.flow import relative_to_best_found


def summary_energies(logfiles: list[pathlib.Path], save_dir: pathlib.Path) -> None:
    """Summarizes the energies over multiple sweeps for each solver and benchmark solved.
    The summary will hold the minimum, maximum, average and std values over the sweep.

    @type logfiles: list[pathlib.Path]
    @param logfiles: a list of all the log files to summarize.
    @type save_dir: pathlib.Path
    @param save_dir: where to store the data.
    """
    energies = dict()

    for logfile in logfiles:
        solver_name = return_metadata(logfile, "solver")
        model_name = return_metadata(logfile, "model_name")

        if energies.get((solver_name, model_name)) is None:
            energies[(solver_name, model_name)] = []

        energy = return_metadata(logfile, "solution_energy")
        energies[(solver_name, model_name)].append(energy)

    header = "min max avg std"
    for (solver_name, model_name), all_energies in energies.items():
        summary = np.array([[np.min(all_energies), np.max(all_energies), np.mean(all_energies), np.std(all_energies)]])
        save_path = save_dir / f"{solver_name}_{model_name}_summary.csv"
        np.savetxt(save_path, summary, fmt="%.2f", header=header)


def box_plot_energies_logfiles(
    logfiles: list[pathlib.Path], best_found: float, save_dir: pathlib.Path, discriminate_by: str | None = None
) -> None:
    """Generates a boxplot from the final energy obtained from a list of logfiles.

    @type logfiles: list[pathlib.Path]
    @param logfiles: the list of logfiles to plot from.
    @type best_found: float
    @param best_found: best found energy to plot as a reference.
    @type save_dir: pathlib.Path
    @param save_dir: the save directory.
    @type discriminate_by: str | None
    @param discriminate_by: to discriminate the colors by. Defaults to None.
    """
    data = get_metadata_from_logfiles(
        logfiles, discriminate_by if discriminate_by is not None else "num_iterations", "solution_energy"
    )

    df = []
    for solver_name, info in data.items():
        for x_dat, y_dat in info.items():
            if discriminate_by is not None:
                df.append(pd.DataFrame({"solver": solver_name, discriminate_by: x_dat, "energy": y_dat}))
            else:
                df.append(pd.DataFrame({"solver": solver_name, "energy": y_dat}))
    df = pd.concat(df)

    plt.figure()
    if discriminate_by is not None:
        sns.boxplot(data=df, x="solver", y="energy", hue=discriminate_by)
    else:
        sns.boxplot(data=df, x="solver", y="energy")
    plt.axhline(y=best_found, color="k", linestyle="--", label=f"Best found: {best_found}")
    plt.legend()
    plt.savefig(save_dir / "boxplot_energies.png", bbox_inches="tight")
    plt.close()


def _per_trial_metric(ans: Ans, solver: str, problem: str) -> np.ndarray:
    """Return the per-trial quality metric for a single Ans. Lower is better.

    For MIMO this is BER per trial; for every other problem it is the gap to the
    best known energy, |E - E*| / |E*|.
    """
    if problem == "MIMO":
        return np.asarray(ans.ber_of_trials[solver], dtype=float)
    energies = np.asarray(ans.energies[solver], dtype=float)
    return relative_to_best_found(energies, ans.best_found)


def _metric_axis_label(problem: str) -> str:
    return "BER" if problem == "MIMO" else "Gap to best known: |E - E*| / |E*|"


def ans_to_metric_df(
    ans_by_label: dict[Any, Ans | list[Ans]],
    label_name: str,
    problem: str,
    solvers: list[str] | None = None,
) -> pd.DataFrame:
    """Convert run results into a long-form DataFrame with one row per trial.

    @type ans_by_label: dict[Any, Ans | list[Ans]]
    @param ans_by_label: maps a label (e.g. parameter value, solver_type, difficulty) to
        a single Ans or a list of Ans (one per benchmark).
    @type label_name: str
    @param label_name: column name to use for the label key.
    @type problem: str
    @param problem: problem type. Selects gap vs. BER as the per-trial metric.
    @type solvers: list[str] | None
    @param solvers: which solvers to extract. Defaults to all solvers in each Ans config.
    @rtype: pd.DataFrame
    @return: DataFrame with columns: C{label_name}, "benchmark", "solver", "metric".
    """
    rows = []
    for label, ans_or_list in ans_by_label.items():
        ans_list = ans_or_list if isinstance(ans_or_list, list) else [ans_or_list]
        for ans in ans_list:
            for solver in solvers or ans.config.solvers:
                for value in _per_trial_metric(ans, solver, problem):
                    rows.append({
                        label_name: label,
                        "benchmark": ans.benchmark,
                        "solver": solver,
                        "metric": float(value),
                    })
    return pd.DataFrame(rows)


def box_plot_metric(
    df: pd.DataFrame,
    x: str,
    problem: str,
    hue: str | None = None,
    title: str = "",
    save_path: pathlib.Path | None = None,
) -> None:
    """Log-y box plot of a per-trial quality metric (gap or BER).

    @type df: pd.DataFrame
    @param df: long-form DataFrame containing at least column C{x} and column "metric".
    @type x: str
    @param x: column name to place on the x-axis.
    @type problem: str
    @param problem: problem type. Used only to set the y-axis label.
    @type hue: str | None
    @param hue: optional column name to group/color by within each x.
    @type title: str
    @param title: figure title.
    @type save_path: pathlib.Path | None
    @param save_path: where to save the figure. If None, the figure is closed without saving.
    """
    plt.figure()
    sns.boxplot(data=df, x=x, y="metric", hue=hue)
    plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel(_metric_axis_label(problem))
    if title:
        plt.title(title)
    if hue is not None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def box_plot_energies_loop(
    ans_data: dict[Any, Ans],
    base_ans: Ans,
    parameter_values: list[Any],
    parameter_name: str,
    problem: str,
    best_found: float | None,
    save_folder: pathlib.Path,
    fig_name: str,
):
    """Box plot of solution quality across a parameter sweep, hue=solver.

    Y-axis is the gap to best known (or BER for MIMO) on a log scale.

    @type ans_data: dict[Any, Ans]
    @param ans_data: results keyed by parameter value.
    @type base_ans: Ans
    @param base_ans: result for the base run (parameter turned off).
    @type parameter_values: list[Any]
    @param parameter_values: parameter values explored in the sweep.
    @type parameter_name: str
    @param parameter_name: the swept parameter — used as the x-axis label.
    @type problem: str
    @param problem: problem type — selects metric (BER vs. gap) and axis label.
    @type best_found: float | None
    @param best_found: kept for signature compatibility; metric is derived per Ans.
    @type save_folder: pathlib.Path
    @param save_folder: figure save root (figure goes to C{save_folder/figures/<fig_name>}).
    @type fig_name: str
    @param fig_name: figure file name.
    """
    del best_found  # metric is derived per Ans

    ans_by_label: dict[Any, Ans] = {"Base": base_ans}
    for value in parameter_values:
        ans_by_label[str(value)] = ans_data[value]

    df = ans_to_metric_df(ans_by_label, label_name=parameter_name, problem=problem)
    box_plot_metric(
        df,
        x=parameter_name,
        problem=problem,
        hue="solver",
        title=f"Solution quality across {parameter_name} values - {problem} problem",
        save_path=save_folder / f"figures/{fig_name}",
    )


def histogram_energies_loop(
    ans_data: dict[Any:Ans],
    ans_base: Ans,
    parameter_values: list[Any],
    parameter_name: str,
    problem: str,
    best_found: float | None,
    fig_name: str,
    save_folder: pathlib.Path,
):
    """Plots a histogram for all the solvers on the different values of C{parameter_name}.

    @type ans_data: dict[Any, Ans]
    @param ans_data: a dictionary containing the answer data for all the different C{parameter_values}.
    @type ans_base: Ans
    @param ans_base: the answer data for the base run with C{parameter_name} turned off.
    @type parameter_values: list[Any]
    @param parameter_values: the list of all the parameter values tested.
    @type parameter_name: str
    @param parameter_name: the name of the parameter.
    @type problem: str
    @param problem: the problem that was tested.
    @type best_found: float | None
    @param best_found: the best found energy to plot as reference.
    @type fig_name: str
    @param fig_name: name of the figure to save.
    @type save_folder: pathlib.Path
    @param save_folder: folder where the figure needs to be saved.
    """
    solvers = ans_base.config.solvers
    for solver in solvers:
        if problem == "MIMO":
            energies_base = []
            for trial in range(ans_base.config.nb_trials):
                energies_base.append(ans_base.MIMO[trial].lowest_energy[solver])
        else:
            energies_base = ans_base.energies[solver]
        plt.figure()
        plt.hist(
            energies_base,
            bins=15,
            alpha=0.7,
            edgecolor="black",
            label=f"Base run: best energy = {np.min(energies_base):.2f}, avg energy: {np.mean(energies_base):.2f}",
        )
        for value in parameter_values:
            if problem == "MIMO":
                energies = []
                for trial in range(ans_base.config.nb_trials):
                    energies.append(ans_data[value].MIMO[trial].lowest_energy[solver])
            else:
                energies = ans_data[value].energies[solver]
            plt.hist(
                energies,
                bins=15,
                alpha=0.7,
                edgecolor="black",
                label=f"{parameter_name} = {value}: best energy = {np.min(energies):.2f}, avg energy = {
                    np.mean(energies):.2f}",
            )

        if best_found is not None:
            if best_found < 0.0:
                plt.axvline(
                    0.9 * best_found,
                    color="k",
                    linestyle="-.",
                    label=f"90% Best known: {0.9 * best_found}",
                )
            elif best_found > 0.0:
                plt.axvline(
                    1.1 * best_found,
                    color="k",
                    linestyle="-.",
                    label=f"90% Best known: {1.1 * best_found}",
                )
            plt.axvline(best_found, color="k", linestyle="--", label=f"Best known: {best_found}")

        plt.title(f"Energy distribution for different {parameter_name} values - {problem} problem")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(save_folder / f"figures/{fig_name}", bbox_inches="tight")
        plt.close()


def pareto_curve_loop(
    ans_data: dict[str : dict[Any : list[Ans]]],
    parameter_name: str,
    parameter_values: list[Any],
    problems: list[str],
    save_folder: pathlib.Path,
    fig_name: str,
):
    """Plots the pareto curve for a parameter from a solver over different benchmarks.

    @type energy_data: dict[str, dict[Any, list]]
    @param energy_data: a dictionary containing the energy data
        for all the different benchmarks.
    @type parameter_name: str
    @param parameter_name: the name of the parameter. This will be put on the x-axis.
    @type parameter_values: list[Any]
    @param parameter_values: the list of all the parameter values tested.
    @type problems: list[str]
    @param problems: the different benchmarks tested.
    @type best_found: dict[str, float]
    @param best_found: a dictionary with the best found energies for every benchmark.
    @type save_folder: pathlib.Path
    @param save_folder: where to save the figure.
    @type fig_name: str
    @param fig_name: name of the figure to save.
    @type solver: str
    @param solver: the solver to plot the pareto curve for.
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:grey"]
    error_colors = ["darkblue", "chocolate", "darkgreen", "maroon"]
    bar_width = 0.4
    parameter_values.sort()
    x = np.arange(0, (4 * bar_width) * len(parameter_values), bar_width * 4)
    for solver in ans_data[problems[0]][parameter_values[0]][0].config.solvers:
        plt.figure()
        fig, ax = plt.subplots()
        # ax = fig.get_axes()[0]
        # ax2 = ax.twinx()
        for ind, problem in enumerate(problems):
            energies_avg = {val: 0.0 for val in parameter_values}
            energies_std = {val: 0.0 for val in parameter_values}
            iterations_avg = {val: 0 for val in parameter_values}
            for val, ans_list in ans_data[problem].items():
                # Store the energies as a relative error to the best found
                energies = np.array([])
                iterations = []
                for ans in ans_list:
                    energies = np.append(
                        energies, relative_to_best_found(np.array(ans.energies[solver]), ans.best_found)
                    )
                    iterations.append(ans.total_iteration_count[solver])

                mean = np.mean(energies)
                energies_avg[val] = mean
                energies_std[val] = np.std(energies)/mean
                iterations_avg[val] = gmean(iterations)
            energies_avg = np.array([energies_avg[val] for val in parameter_values])
            energies_std = np.array([energies_std[val] for val in parameter_values])
            iterations_avg = [iterations_avg[val] for val in parameter_values]
            if problem != "MIMO":
                ax.plot(
                    x,
                    energies_avg,
                    color=colors[ind],
                    marker="o",
                    label=str(problem),
                )
                ax.fill_between(
                    x,
                    energies_avg-energies_std,
                    energies_avg+energies_std,
                    color=error_colors[ind],
                )
                for ind, en in enumerate(energies_avg):
                    ax.text(x[ind], en+en/10, str(iterations_avg[ind]))
            # else:
            #     ax2.errorbar(
            #         x,
            #         energies_avg,
            #         yerr=energies_std,
            #         color=colors[ind],
            #         linestyle="--",
            #         marker="*",
            #         label=str(problem),
            #     )
            #     ax2.set_ylabel("Bit Error Rate", color=colors[ind], fontsize=15)
        ax.set_yscale("log")
        # ax2.set_yscale("log")
        ax.set_ylim(1e-5, 1e5)
        # ax2.set_ylim(1e-4, 1)
        ax.set_xticks(x, [str(val) for val in parameter_values])
        ax.set_xlabel(parameter_name, fontsize=15)
        ax.set_ylabel("Relative distance to best found energy", fontsize=15)
        ax.set_title(f"Pareto curve for different {parameter_name} values - {solver} solver", fontsize=15)
        handles1, labels1 = ax.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        leg = ax.legend(handles1, labels1, fontsize=15, loc="upper left")
        leg.set_zorder(100)
        fig.savefig(save_folder / f"{fig_name}_{solver}.pdf", bbox_inches="tight", dpi=600)
        plt.close()
