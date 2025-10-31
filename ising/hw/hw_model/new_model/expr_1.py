import logging
from pathlib import Path
import yaml
from sachi import sachi_hw_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import copy

def plot_mismatches_in_bar_chart(
    cycles_breakdown_in_list: list,
    mismatch_in_list: list,
    req_sram_size_in_list: list,
    label_in_list: list,
    benchmark_name_in_list: list,
    title: str | None = None,
    component_list: list = [],
    component_tag_list: list = [],
    log_scale: bool = True
    ):
    """
    plot the results breakdown in bar chart
    :param cycles_breakdown_in_list: cycles [ns] in list, each element is a list
    :param mismatch_in_list: mismatch [%] in list, each element is a list
    :param req_sram_size_in_list: required SRAM size [KB] in list, each element is a list
    :param label_in_list: label for each data
    :param benchmark_name_in_list: benchmark name shown on x axis
    :param title: figure title
    :param component_list: list of components for breakdown
    :param component_tag_list: list of component tags for breakdown
    :param log_scale: whether to use log scale for y axis
    """
    colors = {
        "mac": '#45B7D1',  # MAC (MACs)
        "add": '#FFA07A',  # ADD (Adds)
        "comp": '#98D8C8',  # COMP (COMPs)
        "spin_updating": '#F7DC6F',  # SU (Spin Updating)
        "sram": '#4ECDC4',  # L1 (On-chip Memory)
        "dram": '#FF6B6B',  # DRAM (Off-chip Memory)
    }
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    x = list(range(len(cycles_breakdown_in_list)))
    width = 0.25
    details = cycles_breakdown_in_list
    base = np.zeros(len(details))
    for component_idx in range(len(component_list)):
        component = component_list[component_idx]
        for case in details:
            if component not in case:
                case[component] = 0
        breakdown = [case[component] for case in details]
        if component == "mac":
            # plot the macro-level reference
            ax[0].bar(
                [i + width * 0 for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black",
            )
        ax[0].bar(
            [i + width * 1 for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black"
        )
        base += breakdown
    ax0_right = ax[0].twinx()
    ax0_right.plot(
        [i + width * 0.5 for i in x], mismatch_in_list, linestyle="--", color="black", marker="s", markersize=8,
        markerfacecolor="#B32828", markeredgecolor="black"
    )
    # annotate the mismatch values
    for i in range(len(x)):
        ax0_right.annotate(
            f"{mismatch_in_list[i]:.0f}x",
            xy=(i + width * 0.5, mismatch_in_list[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,
            color="#B32828"
        )

    ax[1].bar(
        [i + width * 0 for i in x], req_sram_size_in_list, width=width, color=colors["sram"], edgecolor="black"
    )
    ax[1].hlines(
        y=160, xmin=-1, xmax=len(x)-1 + width * len(label_in_list), colors='black', linestyles='dashed', label='SACHI SRAM Size'
    )

    # set the x, y label
    ax[0].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[0].set_ylabel("Cycles to Solution [cc]", fontsize=15, weight="normal")
    ax0_right.tick_params(axis="y", colors="#B32828")
    ax0_right.set_ylabel("System/Macro Perf. Gap", fontsize=15, weight="normal", color="#B32828")
    ax[1].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[1].set_ylabel("Required SRAM Size [KB]", fontsize=15, weight="normal")
    # set the title
    if title is not None:
        ax[0].set_title(title, fontsize=15)
        ax[1].set_title(title, fontsize=15)
    # set the x tick labels
    ax[0].set_xticks([i + width / 2 for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    # increase x/y tick font size
    plt.setp(ax[0].get_xticklabels(), fontsize=15)
    plt.setp(ax[1].get_xticklabels(), fontsize=15)
    plt.setp(ax[0].get_yticklabels(), fontsize=15)
    plt.setp(ax[1].get_yticklabels(), fontsize=15)
    plt.setp(ax0_right.get_yticklabels(), fontsize=15)
    ax[1].set_xticks([i + width * 0 for i in x])
    ax[1].set_xticklabels(benchmark_name_in_list)
    # create custom legend handles: one for component colors and one for encoding hatch styles
    comp_labels = component_tag_list if component_tag_list else component_list
    # color handles (components)
    color_handles = [Patch(facecolor=colors[comp], edgecolor='black', label=comp_labels[idx])
                     for idx, comp in enumerate(component_list)]

    # Add legends to the left subplot (ax[0]). Use two separate legend objects.
    legend_comp = ax[0].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1), fontsize=15, title_fontsize=15)
    # keep the first legend visible
    ax[0].add_artist(legend_comp)
    # set the y limits
    ax[0].set_ylim(1e5, 1e9)
    ax[1].set_xlim(left=-0.5)

   # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    # ax[0].grid()
    # ax[0].set_axisbelow(True)
    # ax[1].grid()
    # ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/gap_{title}.png", dpi=300)
    logging.warning(f"Saved breakdown figure to ./outputs/gap_{title}.png")

if __name__ == "__main__":
    logging_format = ("%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    hw_model_org = yaml.safe_load(Path.open("./inputs/hardware/sachi.yaml"))
    workload_org = yaml.safe_load(Path.open("./inputs/workload/mc_500.yaml"))
    mapping_org = yaml.safe_load(Path.open("./inputs/mapping/sachi.yaml"))
    component_list = ["mac", "spin_updating", "sram", "dram"]
    component_tag_list = ["MAC", "SU", "SRAM", "DRAM"]
    pb_size_pool = [100, 400, 800, 2000, 4000]
    label_in_list = ["neighbor"]
    benchmark_name_in_list = [f"{pb_size}" for pb_size in pb_size_pool]
    
    # general settings
    weight_shared_precision = 16
    with_bias = True
    problem_specific_weight = True
    sram_size_in_KB = 160
    num_macros = 16

    for aver_density in [0.015]:
        cycle_breakdown_in_list = [[],[],[]]
        req_sram_size_in_list = [[],[],[]]
        mismatch_in_list = [[],[],[]]
        title = f"density_{aver_density:.0%}_precision_{weight_shared_precision}b"
        for pb_size in pb_size_pool:
            for encoding_idx in range(len(label_in_list)):
                encoding = label_in_list[encoding_idx]
                hw_model = copy.deepcopy(hw_model_org)
                workload = copy.deepcopy(workload_org)
                mapping = copy.deepcopy(mapping_org)
                workload["loop_sizes"] = [pb_size, pb_size]
                workload["operand_precision"]["W"] = weight_shared_precision
                workload["operand_precision"]["H"] = weight_shared_precision
                workload["average_degree"] = aver_density * pb_size
                workload["with_bias"] = with_bias
                workload["problem_specific_weight"] = problem_specific_weight
                hw_model["operational_array"]["encoding"] = encoding
                hw_model["memories"]["sram_160KB"]["size"] = sram_size_in_KB * 1024 * 8  # in bits
                hw_model["operational_array"]["sizes"] = [1, 100, num_macros]
                # simulation
                cme = sachi_hw_model(hw_model, workload, mapping)
                # collect results
                cycles_to_solution = cme["cycles_to_solution"]
                energy_to_solution = cme["energy_to_solution"]
                cycle_breakdown_in_list[encoding_idx].append(cme["latency_breakdown_plot"])
                req_sram_size_in_list[encoding_idx].append(cme["req_sram_size_bit"]/8/1024)  # in KB
                mismatch = cycles_to_solution / cme["latency_breakdown_plot"]["mac"]
                mismatch_in_list[encoding_idx].append(mismatch)
        # plot the results
        plot_mismatches_in_bar_chart(
            cycles_breakdown_in_list=cycle_breakdown_in_list[0],
            mismatch_in_list=mismatch_in_list[0],
            req_sram_size_in_list=req_sram_size_in_list[0],
            label_in_list=label_in_list,
            benchmark_name_in_list=benchmark_name_in_list,
            title=title,
            component_list=component_list,
            component_tag_list=component_tag_list,
            log_scale=True
        )