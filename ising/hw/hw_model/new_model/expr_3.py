import logging
import yaml
from sachi import sachi_hw_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import copy
import math
import tqdm
from get_cacti_cost import get_cacti_cost

def plot_perf_ratio_in_curve(
    latency_mismatch_in_list: list,
    energy_mismatch_in_list: list,
    targeted_annotation_idx: int,
    benchmark_name_in_list: list,
    title: str | None = None,
    log_scale: bool = True,
    ):
    """
    plot the performance ratio in curve
    :param latency_mismatch_in_list: latency mismatch [%] in list, each element is a list
    :param energy_mismatch_in_list: energy mismatch [%] in list, each element is a list
    :param targeted_annotation_idx: only annotate the data at this index
    :param benchmark_name_in_list: label for each data
    :param title: figure title
    :param log_scale: whether to use log scale for y axis
    """
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))

    x = list(range(len(latency_mismatch_in_list[0])))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p"]
    for idx in range(len(latency_mismatch_in_list)):
        ax[0].plot(
            [i for i in x],
            latency_mismatch_in_list[idx],
            marker=markers[idx],
            label=label_in_list[idx],
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        ax[1].plot(
            [i for i in x],
            energy_mismatch_in_list[idx],
            marker=markers[idx],
            label=label_in_list[idx],
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        # if idx == targeted_annotation_idx:
        if False:
            # add annotation for each point
            for i in range(len(x)):
                ax[0].annotate(
                    f"{latency_mismatch_in_list[idx][i]:.0f}x",
                    (i, latency_mismatch_in_list[idx][i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=12,
                    color="black"
                )
                ax[1].annotate(
                    f"{energy_mismatch_in_list[idx][i]:.0f}x",
                    (i, energy_mismatch_in_list[idx][i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=12,
                    color="black"
                )

    # set the x tick labels
    ax[0].set_xticks([i for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    ax[1].set_xticks([i for i in x])
    ax[1].set_xticklabels(benchmark_name_in_list)

    # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

    # increase x/y tick font size
    plt.setp(ax[0].get_xticklabels(), fontsize=15)
    plt.setp(ax[0].get_yticklabels(), fontsize=15)
    plt.setp(ax[1].get_xticklabels(), fontsize=15)
    plt.setp(ax[1].get_yticklabels(), fontsize=15)

    # add the x, y labels
    ax[0].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[0].set_ylabel("System/Macro\nLatency Ratio", fontsize=15, weight="normal")
    ax[1].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[1].set_ylabel("System/Macro\nEnergy Ratio", fontsize=15, weight="normal")

    # add legend
    ax[0].legend(fontsize=15, loc="upper right")
    ax[1].legend(fontsize=15, loc="upper right")

    # add grid and put grid below axis
    ax[0].grid(which="both")
    ax[0].set_axisbelow(True)
    ax[1].grid(which="both")
    ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/expr_2_perf_ratio_{title}.png", dpi=300)
    plt.savefig(f"./outputs/expr_2_perf_ratio_{title}.svg", dpi=300)
    logging.warning(f"Saved performance ratio figure to ./outputs/expr_2_perf_ratio_{title}.png")

def plot_results_breakdown_in_bar_chart(
    cycles_breakdown_in_list: list,
    label_in_list: list,
    benchmark_name_in_list: list,
    energy_breakdown_in_list: list,
    area_breakdown_in_list: list,
    title: str | None = None,
    component_list: list = [],
    component_tag_list: list = [],
    cycles_in_list: list = [],
    energy_in_list: list = [],
    log_scale: bool = True,
    topsmm2_in_list: list = [],
    topsw_in_list: list = [],
    disable_right_axis: bool = False,
    showing_legend: bool = True,
    ):
    """
    plot the results breakdown in bar chart
    :param cycles_breakdown_in_list: cycles [ns] in list, each element is a list
    :param label_in_list: label for each data
    :param benchmark_name_in_list: benchmark name shown on x axis
    :param energy_breakdown_in_list: energy [pJ] in list, each element is a list
    :param area_breakdown_in_list: area [mm2] in list, each element is a list
    :param title: figure title
    :param component_list: list of components for breakdown
    :param component_tag_list: list of component tags for breakdown
    :param cycles_in_list: total cycles [ns] in list, each element is a list [not used here]
    :param energy_in_list: total energy [pJ] in list, each element is a list [not used here]
    :param log_scale: whether to use log scale for y axis
    :param topsmm2_in_list: list of TOPS (MM2) for each data
    :param topsw_in_list: list of TOPS (W) for each data
    :param disable_right_axis: whether to disable the right y axis
    :param showing_legend: whether to show the legend
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
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    x = list(range(len(cycles_breakdown_in_list[0])))
    width = 0.125
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
                [i + width * idx for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black",
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
                [i + width * idx for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black",
            )
            base += breakdown

    for idx in range(len(area_breakdown_in_list)):
        details = area_breakdown_in_list[idx]
        base = np.zeros(len(details))
        for component_idx in range(len(component_list)):
            component = component_list[component_idx]
            for case in details:
                if component not in case:
                    case[component] = 0
            breakdown = [case[component] for case in details]
            ax[2].bar(
                [i + width * idx for i in x], breakdown, bottom=base, width=width, color=colors[component], edgecolor="black",
            )
            base += breakdown

    # area_x = list(range(len(label_in_list)))
    # base = np.zeros(len(label_in_list))
    # for idx in range(len(component_list)):
    #     component = component_list[idx]
    #     if component == "dram":
    #         continue  # skip DRAM for area breakdown
    #     try:
    #         breakdown = [case[0][component] for case in area_breakdown_in_list]
    #     except KeyError:
    #         breakpoint()
    #     ax[2].bar(
    #             area_x, breakdown, bottom=base, width=0.5, color=colors[component], edgecolor="black", label=component_tag_list[component_idx]
    #         )
    #     base += breakdown

    if not disable_right_axis:
        # plot the topsmm2 and topsw on the right y axis
        ax0_right = ax[0].twinx()
        ax1_right = ax[1].twinx()
        markers = ["o", "s", "D", "^", "v", "<", ">", "p"]
        for idx in range(len(topsmm2_in_list)):
            ax0_right.scatter(
                [i + width * idx for i in x],
                topsmm2_in_list[idx],
                edgecolors="#B32828",
                facecolors="black",
                marker=markers[idx],
            )
            ax1_right.scatter(
                [i + width * idx for i in x],
                topsw_in_list[idx],
                edgecolors="#B32828",
                facecolors="black",
                marker=markers[idx],
            )

    # set the x, y label
    ax[0].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[0].set_ylabel("Cycles to Solution [cc]", fontsize=15, weight="normal")
    ax[1].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[1].set_ylabel("Energy to Solution [pJ]", fontsize=15, weight="normal")
    ax[2].set_xlabel("Problem Size", fontsize=15, weight="normal")
    ax[2].set_ylabel("Area [mm$^2$]", fontsize=15, weight="normal")
    if not disable_right_axis:
        ax0_right.set_ylabel("TOP/s/mm$^2$", fontsize=15, weight="normal", color="#B32828")
        ax1_right.set_ylabel("TOP/s/W", fontsize=15, weight="normal", color="#B32828")
        ax0_right.tick_params(axis="y", colors="#B32828")
        ax1_right.tick_params(axis="y", colors="#B32828")

    if not disable_right_axis:
        # annotate the topsmm2 and topsw values
        # first normalize the topsmm2 and topsw
        topsmm2_copy = copy.deepcopy(topsmm2_in_list)
        topsw_copy = copy.deepcopy(topsw_in_list)
        for idx in range(len(topsmm2_in_list[0])):
            topsmm2_encoding_1 = topsmm2_in_list[0][idx]
            topsmm2_encoding_2 = topsmm2_in_list[1][idx]
            topsmm2_encoding_3 = topsmm2_in_list[2][idx]
            topsmm2_min = min(topsmm2_encoding_1, topsmm2_encoding_2, topsmm2_encoding_3)
            topsmm2_copy[0][idx] /= topsmm2_min
            topsmm2_copy[1][idx] /= topsmm2_min
            topsmm2_copy[2][idx] /= topsmm2_min
            topsw_encoding_1 = topsw_in_list[0][idx]
            topsw_encoding_2 = topsw_in_list[1][idx]
            topsw_encoding_3 = topsw_in_list[2][idx]
            topsw_min = min(topsw_encoding_1, topsw_encoding_2, topsw_encoding_3)
            topsw_copy[0][idx] /= topsw_min
            topsw_copy[1][idx] /= topsw_min
            topsw_copy[2][idx] /= topsw_min

        for idx in range(len(topsmm2_in_list)):
            for i in range(len(topsmm2_in_list[idx])):
                ax0_right.annotate(
                    f"{topsmm2_copy[idx][i]:.0f}x",
                    (i + width * idx, topsmm2_in_list[idx][i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=15,
                    color="black"
                )
                ax1_right.annotate(
                    f"{topsw_copy[idx][i]:.0f}x",
                    (i + width * idx, topsw_in_list[idx][i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=15,
                    color="black"
                )

    # set the x tick labels
    ax[0].set_xticks([i + width * (len(cycles_breakdown_in_list)-1)/2 for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    ax[1].set_xticks([i + width * (len(cycles_breakdown_in_list)-1)/2 for i in x])
    ax[1].set_xticklabels(benchmark_name_in_list)
    # ax[2].set_xticks([i for i in area_x])
    # ax[2].set_xticklabels([label[5:] for label in label_in_list])
    ax[2].set_xticks([i + width * (len(cycles_breakdown_in_list)-1)/2 for i in x])
    ax[2].set_xticklabels(benchmark_name_in_list)

    if showing_legend:
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
        legend_comp = ax[0].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1), fontsize=15, title_fontsize=15, ncol=2)
        ax[0].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1), fontsize=15, title_fontsize=15)
        # keep the first legend visible
        ax[0].add_artist(legend_comp)

        # Mirror legends on the right subplot (ax[1]) for consistency
        legend_comp_r = ax[1].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1), fontsize=15, title_fontsize=15, ncol=2)
        ax[1].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1), fontsize=15, title_fontsize=15)
        ax[1].add_artist(legend_comp_r)

        # add legends to the area subplot (ax[2])
        ax[2].legend(handles=color_handles, title='Component', loc='upper right', fontsize=15, title_fontsize=15, ncol=2)

    # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        # ax[2].set_yscale("log")
        if not disable_right_axis:
            ax0_right.set_yscale("log")
            ax1_right.set_yscale("log")

    # increase x/y tick font size
    plt.setp(ax[0].get_xticklabels(), fontsize=15)
    plt.setp(ax[1].get_xticklabels(), fontsize=15)
    plt.setp(ax[0].get_yticklabels(), fontsize=15)
    plt.setp(ax[1].get_yticklabels(), fontsize=15)
    plt.setp(ax[2].get_xticklabels(), fontsize=15)
    plt.setp(ax[2].get_yticklabels(), fontsize=15)
    if not disable_right_axis:
        plt.setp(ax0_right.get_yticklabels(), fontsize=15)
        plt.setp(ax1_right.get_yticklabels(), fontsize=15)

    # set the y range
    # ax[0].set_ylim(1e3, 1e11)
    ax[1].set_ylim(1e4, 1e12)
    ax[2].set_ylim(0, 12)

    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add grid and put grid below axis
    ax[0].grid(which="both")
    ax[0].set_axisbelow(True)
    ax[1].grid(which="both")
    ax[1].set_axisbelow(True)
    ax[2].grid(which="both")
    ax[2].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/expr_3_{title}.png", dpi=300)
    plt.savefig(f"./outputs/expr_3_{title}.svg", dpi=300)
    logging.warning(f"Saved breakdown figure to ./outputs/expr_3_{title}.png")


if __name__ == "__main__":
    logging_format = ("%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    hw_model_org = yaml.safe_load(open("./inputs/hardware/sachi.yaml", 'r'))
    workload_org = yaml.safe_load(open("./inputs/workload/mc_500.yaml", 'r'))
    mapping_org = yaml.safe_load(open("./inputs/mapping/sachi.yaml", 'r'))
    component_list = ["mac", "spin_updating", "sram", "dram"]
    component_tag_list = ["MAC", "L1", "L2", "DRAM"]
    # experiment: sweep different problem sizes and encoding methods
    pb_pool = [
        # pb_size, degree density, weight precision, with bias, problem specific weight
        [200, 0.015, 1, False, True], # MaxCut
        [4000, 0.015, 1, False, True], # MaxCut
        [200, 2*((200**0.5)-1)/199, 16, True, True], # TSP
        [8000, 2*((8000**0.5)-1)/7999, 16, True, True], # TSP
        [200, 28/100, 3, True, False], # Sudoku
        [729, 28/729, 3, True, False], # Sudoku
        [64, 1, 16, True, False], # MIMO
        [256, 1, 16, True, False], # MIMO
    ]
    benchmark_name_in_list = [f"{pb_spec[0]}" for pb_spec in pb_pool]
    d2_in_list = [1, 8, 32, 64, 100, 256, 512]
    # d2_in_list = [100, 256]
    
    # general settings
    sram_size_in_KB = 160
    num_macros = 16
    cim_depth = 80
    encoding = "neighbor"
    label_in_list = [f"MAC: {d2 * num_macros}" for d2 in d2_in_list]


    cycles_in_list = [[] for _ in range(len(d2_in_list))]
    energy_in_list = [[] for _ in range(len(d2_in_list))]
    cycle_breakdown_in_list = [[] for _ in range(len(d2_in_list))]
    energy_breakdown_in_list = [[] for _ in range(len(d2_in_list))]
    area_breakdown_in_list = [[] for _ in range(len(d2_in_list))]
    req_sram_size_in_list = [[] for _ in range(len(d2_in_list))]
    tops_in_list = [[] for _ in range(len(d2_in_list))]
    topsw_in_list = [[] for _ in range(len(d2_in_list))]
    topsmm2_in_list = [[] for _ in range(len(d2_in_list))]
    tops_macro_in_list = [[] for _ in range(len(d2_in_list))]
    topsw_macro_in_list = [[] for _ in range(len(d2_in_list))]
    topsmm2_macro_in_list = [[] for _ in range(len(d2_in_list))]
    latency_mismatch_in_list = [[] for _ in range(len(d2_in_list))]
    energy_mismatch_in_list = [[] for _ in range(len(d2_in_list))]
    title = f"MAC_experiment_{encoding}_encoding"
    pbar = tqdm.tqdm(total=len(pb_pool)*len(d2_in_list))
    for pb_spec in pb_pool:
        for d2_idx in range(len(d2_in_list)):
            d2 = d2_in_list[d2_idx]
            hw_model = copy.deepcopy(hw_model_org)
            workload = copy.deepcopy(workload_org)
            mapping = copy.deepcopy(mapping_org)
            pb_size, aver_density, weight_shared_precision, with_bias, problem_specific_weight = pb_spec
            if encoding == "coordinate":
                bit_per_weight = weight_shared_precision + math.log2(pb_size)
            elif encoding == "neighbor":
                bit_per_weight = weight_shared_precision + 1
            else:  # full-matrix
                bit_per_weight = weight_shared_precision
            # setup workload and hw_model
            workload["loop_sizes"] = [pb_size, pb_size]
            workload["operand_precision"]["W"] = weight_shared_precision
            workload["operand_precision"]["H"] = weight_shared_precision
            workload["average_degree"] = aver_density * pb_size
            workload["with_bias"] = with_bias
            workload["problem_specific_weight"] = problem_specific_weight
            hw_model["memories"]["sram_160KB"]["size"] = sram_size_in_KB * 1024 * 8  # in bits
            _, hw_model["memories"]["sram_160KB"]["area"], hw_model["memories"]["sram_160KB"]["r_cost"], hw_model["memories"]["sram_160KB"]["w_cost"] = get_cacti_cost(cacti_path='./cacti/cacti_master', tech_node=0.028,
                                                                            mem_type='sram', mem_size_in_byte=sram_size_in_KB * 1024,
                                                                            bw=hw_model["memories"]["sram_160KB"]["bandwidth"])

            # hw_model["memories"]["cim_memory"]["bandwidth"] = 1024
            hw_model["memories"]["cim_memory"]["bandwidth"] = d2 * bit_per_weight
            hw_model["memories"]["cim_memory"]["size"] = cim_depth * hw_model["memories"]["cim_memory"]["bandwidth"]  # in bits
            _, hw_model["memories"]["cim_memory"]["area"], hw_model["memories"]["cim_memory"]["r_cost"], hw_model["memories"]["cim_memory"]["w_cost"] = get_cacti_cost(cacti_path='./cacti/cacti_master', tech_node=0.028,
                                                                            mem_type='sram', mem_size_in_byte=hw_model["memories"]["cim_memory"]["size"]/8,
                                                                            bw=hw_model["memories"]["cim_memory"]["bandwidth"])
            hw_model["operational_array"]["encoding"] = encoding
            hw_model["operational_array"]["sizes"] = [1, d2, num_macros]
            # linearly scale the mac/add/compare energy according to weight precision, linearly is because it is 1-bit*n-bit a mac logic
            hw_model["operational_array"]["mac_energy"] = hw_model["operational_array"]["mac_energy"] / 8 * weight_shared_precision
            hw_model["operational_array"]["add_energy"] = hw_model["operational_array"]["add_energy"] / 8 * weight_shared_precision
            hw_model["operational_array"]["compare_energy"] = hw_model["operational_array"]["compare_energy"] / 8 * weight_shared_precision
            if encoding == "full-matrix":
                # full-matrix has 0.24x energy due to no decoding, the ratio is extracted from PRIM-CAEFA paper, Fig. 7
                hw_model["operational_array"]["mac_energy"] *= 0.24
                hw_model["operational_array"]["add_energy"] *= 0.24
                hw_model["operational_array"]["compare_energy"] *= 0.24
                hw_model["operational_array"]["mac_area"] *= 0.24
            # simulation
            cme = sachi_hw_model(hw_model, workload, mapping)
            # collect results
            cycles_to_solution = cme["cycles_to_solution"]
            energy_to_solution = cme["energy_to_solution"]
            cycles_in_list[d2_idx].append(cycles_to_solution)
            energy_in_list[d2_idx].append(energy_to_solution)
            cycle_breakdown_in_list[d2_idx].append(cme["latency_breakdown_plot"])
            energy_breakdown_in_list[d2_idx].append(cme["energy_breakdown_plot"])
            area_breakdown_in_list[d2_idx].append(cme["area_breakdown_plot"])
            req_sram_size_in_list[d2_idx].append(cme["req_sram_size_bit"]/8/1024)  # in KB
            tops_in_list[d2_idx].append(cme["tops"])
            topsw_in_list[d2_idx].append(cme["topsw"])
            topsmm2_in_list[d2_idx].append(cme["topsmm2"])
            tops_macro_in_list[d2_idx].append(cme["tops_macro"])
            topsw_macro_in_list[d2_idx].append(cme["topsw_macro"])
            topsmm2_macro_in_list[d2_idx].append(cme["topsmm2_macro"])
            # calculate the latency and energy mismatch compared to macro level
            latency_mismatch = cycles_to_solution / cme["latency_breakdown_plot"]["mac"]
            latency_mismatch_in_list[d2_idx].append(latency_mismatch)
            energy_mismatch = energy_to_solution / cme["energy_breakdown_plot"]["mac"]
            energy_mismatch_in_list[d2_idx].append(energy_mismatch)
            pbar.update(1)
    pbar.close()
    # plot the results
    plot_results_breakdown_in_bar_chart(
        cycles_in_list=cycles_in_list,
        cycles_breakdown_in_list=cycle_breakdown_in_list,
        energy_in_list=energy_in_list,
        energy_breakdown_in_list=energy_breakdown_in_list,
        label_in_list=label_in_list,
        benchmark_name_in_list=benchmark_name_in_list,
        title=f"{title}",
        component_list=component_list,
        component_tag_list=component_tag_list,
        topsmm2_in_list=topsmm2_in_list,
        area_breakdown_in_list=area_breakdown_in_list,
        topsw_in_list=topsw_in_list,
        log_scale=True,
        disable_right_axis=True,
        showing_legend=False,
    )
    plot_perf_ratio_in_curve(
        latency_mismatch_in_list=topsmm2_in_list,
        energy_mismatch_in_list=topsw_in_list,
        targeted_annotation_idx=1, # only annotate neighbor encoding
        benchmark_name_in_list=benchmark_name_in_list,
        title=f"{title}",
        log_scale=True,
    )

