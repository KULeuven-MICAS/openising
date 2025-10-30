import logging
import yaml
from typing import Optional
from sachi import sachi_hw_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import copy

def plot_results_in_bar_chart(
    cycles_in_list: list,
    label_in_list: list,
    benchmark_name_in_list: list,
    energy_in_list: list,
    title: Optional[str] = None,
    component_list: list = [],
    component_tag_list: list = [],
    cycles_breakdown_in_list: list = [],
    energy_breakdown_in_list: list = [],
    log_scale: bool = True
    ):
    """
    plot the results in bar chart
    :param cycles_in_list: cycles [ns] in list, each element is a list
    :param label_in_list: label for each data
    :param benchmark_name_in_list: benchmark name shown on x axis
    :param energy_in_list: energy [pJ] in list, each element is a list
    :param title: figure title
    :param component_list: list of components for breakdown [not used here]
    :param component_tag_list: list of component tags for breakdown [not used here]
    :param cycles_breakdown_in_list: cycles breakdown [ns] in list, each element is a list [not used here]
    :param energy_breakdown_in_list: energy breakdown [pJ] in list, each element is a list [not used here]
    :param log_scale: whether to use log scale for y axis
    """
    colors = [
        '#4cccc5',
        '#f7de6e',
        '#fc9f79',
        '#97d2c2'
    ]
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    x = list(range(len(cycles_in_list[0])))
    width = 0.15
    for idx in range(len(cycles_in_list)):
        ax[0].bar(
        [i + width * idx for i in x], cycles_in_list[idx], width, label=label_in_list[idx], color=colors[idx], edgecolor="black"
        )
        ax[1].bar(
        [i + width * idx for i in x], energy_in_list[idx], width, label=label_in_list[idx], color=colors[idx], edgecolor="black"
        )
    # set the x, y label
    ax[0].set_xlabel("Problem Size", fontsize=12, weight="normal")
    ax[0].set_ylabel("Cycles to Solution [cc]", fontsize=12, weight="normal")
    ax[1].set_xlabel("Problem Size", fontsize=12, weight="normal")
    ax[1].set_ylabel("Energy to Solution [pJ]", fontsize=12, weight="normal")
    # set the title
    if title is not None:
        ax[0].set_title(title)
        ax[1].set_title(title)
    # set the x tick labels
    ax[0].set_xticks([i + width / 2 for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    ax[1].set_xticks([i + width / 2 for i in x])
    ax[1].set_xticklabels(benchmark_name_in_list)
    # set the legend
    ax[0].legend()
    ax[1].legend()
    # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
    # set the y range
    ax[0].set_ylim(1e2, 1e8)
    ax[1].set_ylim(1e3, 1e11)
    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/expr1_{title}.png", dpi=300)

def plot_results_breakdown_in_bar_chart(
    cycles_breakdown_in_list: list,
    label_in_list: list,
    benchmark_name_in_list: list,
    energy_breakdown_in_list: list,
    title: Optional[str] = None,
    component_list: list = [],
    component_tag_list: list = [],
    cycles_in_list: list = [],
    energy_in_list: list = [],
    log_scale: bool = True
    ):
    """
    plot the results breakdown in bar chart
    :param cycles_breakdown_in_list: cycles [ns] in list, each element is a list
    :param label_in_list: label for each data
    :param benchmark_name_in_list: benchmark name shown on x axis
    :param energy_breakdown_in_list: energy [pJ] in list, each element is a list
    :param title: figure title
    :param component_list: list of components for breakdown
    :param component_tag_list: list of component tags for breakdown
    :param cycles_in_list: total cycles [ns] in list, each element is a list [not used here]
    :param energy_in_list: total energy [pJ] in list, each element is a list [not used here]
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
    hatchs = ["x", "//", "oo", "++", "**", "||", "..", "\\\\"]
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    x = list(range(len(cycles_breakdown_in_list[0])))
    width = 0.15
    for idx in range(len(cycles_breakdown_in_list)):
        details = cycles_breakdown_in_list[idx]
        base = np.zeros(len(details))
        for component_idx in range(len(component_list)):
            component = component_list[component_idx]
            for case in details:
                if component not in case:
                    case[component] = 0
            breakdown = [case[component] for case in details]
            
            ax[0].bar(
                [i + width * idx for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black", hatch=hatchs[idx]
            )
            base += breakdown

    for idx in range(len(energy_breakdown_in_list)):
        details = energy_breakdown_in_list[idx]
        base = np.zeros(len(details))
        for component_idx in range(len(component_list)):
            component = component_list[component_idx]
            for case in details:
                if component not in case:
                    case[component] = 0
            breakdown = [case[component] for case in details]
            ax[1].bar(
                [i + width * idx for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black", hatch=hatchs[idx]
            )
            base += breakdown

    # set the x, y label
    ax[0].set_xlabel("Problem Size", fontsize=12, weight="normal")
    ax[0].set_ylabel("Cycles to Solution [cc]", fontsize=12, weight="normal")
    ax[1].set_xlabel("Problem Size", fontsize=12, weight="normal")
    ax[1].set_ylabel("Energy to Solution [pJ]", fontsize=12, weight="normal")
    # set the title
    if title is not None:
        ax[0].set_title(title)
        ax[1].set_title(title)
    # set the x tick labels
    ax[0].set_xticks([i + width / 2 for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    ax[1].set_xticks([i + width / 2 for i in x])
    ax[1].set_xticklabels(benchmark_name_in_list)
    # create custom legend handles: one for component colors and one for encoding hatch styles
    comp_labels = component_tag_list if component_tag_list else component_list
    # color handles (components)
    color_handles = [Patch(facecolor=colors[comp], edgecolor='black', label=comp_labels[idx])
                     for idx, comp in enumerate(component_list)]
    # hatch handles (encodings / labels)
    hatch_handles = []
    for idx, lab in enumerate(label_in_list):
        h = hatchs[idx % len(hatchs)]
        # Rectangle with hatch to show hatch style; use white facecolor so hatch is visible
        hatch_handles.append(Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch=h, label=lab))

    # Add legends to the left subplot (ax[0]). Use two separate legend objects.
    legend_comp = ax[0].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1))
    legend_hatch = ax[0].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1))
    # keep the first legend visible
    ax[0].add_artist(legend_comp)

    # Mirror legends on the right subplot (ax[1]) for consistency
    legend_comp_r = ax[1].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1))
    legend_hatch_r = ax[1].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1))
    ax[1].add_artist(legend_comp_r)
    # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
    # set the y range
    ax[0].set_ylim(1e2, 1e8)
    ax[1].set_ylim(1e3, 1e11)
    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/expr1_bd_{title}.png", dpi=300)


if __name__ == "__main__":
    logging_format = ("%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    hw_model_org = yaml.safe_load(open("./inputs/hardware/sachi.yaml", 'r'))
    workload_org = yaml.safe_load(open("./inputs/workload/mc_500.yaml", 'r'))
    mapping_org = yaml.safe_load(open("./inputs/mapping/sachi.yaml", 'r'))
    component_list = ["mac", "spin_updating", "sram", "dram"]
    component_tag_list = ["MAC", "SU", "SRAM", "DRAM"]
    # experiment: sweep different problem sizes and encoding methods
    pb_size_pool = [100, 400, 800, 2000, 4000]
    label_in_list = ["coordinate", "neighbor", "full-matrix"]
    benchmark_name_in_list = [f"{pb_size}" for pb_size in pb_size_pool]
    
    # general settings
    weight_shared_precision = 1
    with_bias = True
    problem_specific_weight = True
    sram_size_in_KB = 160

    for aver_density in [1, 0.015]:
        cycles_in_list = [[],[],[]]
        energy_in_list = [[],[],[]]
        cycle_breakdown_in_list = [[],[],[]]
        energy_breakdown_in_list = [[],[],[]]
        req_sram_size_in_list = [[],[],[]]
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
                cme = sachi_hw_model(hw_model, workload, mapping)
                cycles_to_solution = cme["cycles_to_solution"]
                energy_to_solution = cme["energy_to_solution"]
                cycles_in_list[encoding_idx].append(cycles_to_solution)
                energy_in_list[encoding_idx].append(energy_to_solution)
                cycle_breakdown_in_list[encoding_idx].append(cme["latency_breakdown_plot"])
                energy_breakdown_in_list[encoding_idx].append(cme["energy_breakdown_plot"])
                req_sram_size_in_list[encoding_idx].append(cme["req_sram_size_bit"]/8/1024)  # in KB
        plot_results_breakdown_in_bar_chart(
            cycles_in_list=cycles_in_list,
            cycles_breakdown_in_list=cycle_breakdown_in_list,
            energy_in_list=energy_in_list,
            energy_breakdown_in_list=energy_breakdown_in_list,
            label_in_list=label_in_list,
            benchmark_name_in_list=benchmark_name_in_list,
            title=title,
            component_list=component_list,
            component_tag_list=component_tag_list,
            log_scale=True
        )

