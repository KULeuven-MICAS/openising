import logging
import math
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
from matplotlib.patches import Patch, Rectangle

def plot_results_breakdown_in_bar_chart(
    cycles_breakdown_in_list: list,
    label_in_list: list,
    benchmark_name_in_list: list,
    energy_breakdown_in_list: list,
    title: str | None = None,
    component_list: list = [],
    component_tag_list: list = [],
    cycles_in_list: list = [],
    energy_in_list: list = [],
    log_scale: bool = True,
    topsmm2_in_list: list = [],
    topsw_in_list: list = [],
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
    :param topsmm2_in_list: list of TOPS (MM2) for each data
    :param topsw_in_list: list of TOPS (W) for each data
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
    ax0_right.set_ylabel("TOP/s/mm$^2$", fontsize=15, weight="normal", color="#B32828")
    ax1_right.set_ylabel("TOP/s/W", fontsize=15, weight="normal", color="#B32828")
    ax0_right.tick_params(axis="y", colors="#B32828")
    ax1_right.tick_params(axis="y", colors="#B32828")

    # annotate the cycle and energy values
    # first normalize the cycles and energy
    cycles_copy = copy.deepcopy(cycles_in_list)
    energy_copy = copy.deepcopy(energy_in_list)
    for idx in range(len(cycles_in_list[0])):
        cycles_case_1 = cycles_in_list[0][idx]
        cycles_case_2 = cycles_in_list[1][idx]
        cycles_case_3 = cycles_in_list[2][idx]
        cycles_case_4 = cycles_in_list[3][idx]
        cycles_min = min(cycles_case_1, cycles_case_2, cycles_case_3, cycles_case_4)
        cycles_copy[0][idx] /= cycles_min
        cycles_copy[1][idx] /= cycles_min
        cycles_copy[2][idx] /= cycles_min
        cycles_copy[3][idx] /= cycles_min
        energy_case_1 = energy_in_list[0][idx]
        energy_case_2 = energy_in_list[1][idx]
        energy_case_3 = energy_in_list[2][idx]
        energy_case_4 = energy_in_list[3][idx]
        energy_min = min(energy_case_1, energy_case_2, energy_case_3, energy_case_4)
        energy_copy[0][idx] /= energy_min
        energy_copy[1][idx] /= energy_min
        energy_copy[2][idx] /= energy_min
        energy_copy[3][idx] /= energy_min

    for idx in range(len(cycles_in_list)):
        for i in range(len(cycles_in_list[idx])):
            ax[0].annotate(
                f"{cycles_copy[idx][i]:.0f}x",
                (i + width * idx, cycles_in_list[idx][i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=15,
                color="black"
            )
            ax[1].annotate(
                f"{energy_copy[idx][i]:.0f}x",
                (i + width * idx, energy_in_list[idx][i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=15,
                color="black"
            )

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

    # for idx in range(len(topsmm2_in_list)):
    #     for i in range(len(topsmm2_in_list[idx])):
    #         ax0_right.annotate(
    #             f"{topsmm2_copy[idx][i]:.0f}x",
    #             (i + width * idx, topsmm2_in_list[idx][i]),
    #             textcoords="offset points",
    #             xytext=(0, 5),
    #             ha="center",
    #             fontsize=15,
    #             color="black"
    #         )
    #         ax1_right.annotate(
    #             f"{topsw_copy[idx][i]:.0f}x",
    #             (i + width * idx, topsw_in_list[idx][i]),
    #             textcoords="offset points",
    #             xytext=(0, 5),
    #             ha="center",
    #             fontsize=15,
    #             color="black"
    #         )

    # set the title
    if title is not None:
        ax[0].set_title(title)
        ax[1].set_title(title)
    # set the x tick labels
    ax[0].set_xticks([i + width for i in x])
    ax[0].set_xticklabels(benchmark_name_in_list)
    ax[1].set_xticks([i + width for i in x])
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
    legend_comp = ax[0].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1), fontsize=15, title_fontsize=15, ncol=2)
    ax[0].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1), fontsize=15, title_fontsize=15)
    # keep the first legend visible
    ax[0].add_artist(legend_comp)

    # Mirror legends on the right subplot (ax[1]) for consistency
    legend_comp_r = ax[1].legend(handles=color_handles, title='Component', loc='upper left', bbox_to_anchor=(0, 1), fontsize=15, title_fontsize=15, ncol=2)
    ax[1].legend(handles=hatch_handles, title='Encoding', loc='upper right', bbox_to_anchor=(1, 1), fontsize=15, title_fontsize=15)
    ax[1].add_artist(legend_comp_r)
    # set the y scale to log scale
    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax0_right.set_yscale("log")
        ax1_right.set_yscale("log")

    # increase x/y tick font size
    plt.setp(ax[0].get_xticklabels(), fontsize=15)
    plt.setp(ax[1].get_xticklabels(), fontsize=15)
    plt.setp(ax[0].get_yticklabels(), fontsize=15)
    plt.setp(ax[1].get_yticklabels(), fontsize=15)
    plt.setp(ax0_right.get_yticklabels(), fontsize=15)
    plt.setp(ax1_right.get_yticklabels(), fontsize=15)

    # set the y range
    ax[0].set_ylim(1e3, 1e11)
    ax[1].set_ylim(1e4, 1e13)
    ax0_right.set_ylim(1e-4, 1e3)
    ax1_right.set_ylim(1e-3, 1e6)
    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    # ax[0].grid()
    # ax[0].set_axisbelow(True)
    # ax[1].grid()
    # ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/expr3_bd_{title}.png", dpi=300)
    logging.warning(f"Saved breakdown figure to ./outputs/expr3_bd_{title}.png")

def analyze_expr_sweep_results(results_file_path, analysis_output_path):
    # Load results from pickle file
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)

    max_200_results = [row for row in results if row[0] == 200 and row[1] == 0.015 and row[2] == 1 and row[3] == False and row[4] == True]
    max_4000_results = [row for row in results if row[0] == 4000 and row[1] == 0.015 and row[2] == 1 and row[3] == False and row[4] == True]
    tsp_200_results = [row for row in results if row[0] == 200 and row[1] == (2*((200**0.5)-1)/199) and row[2] == 16 and row[3] == True and row[4] == True]
    tsp_8000_results = [row for row in results if row[0] == 8000 and row[1] == (2*((8000**0.5)-1)/7999) and row[2] == 16 and row[3] == True and row[4] == True]
    sudoku_200_results = [row for row in results if row[0] == 200 and row[1] == (28/100) and row[2] == 3 and row[3] == True and row[4] == False]
    sudoku_729_results = [row for row in results if row[0] == 729 and row[1] == (28/729) and row[2] == 3 and row[3] == True and row[4] == False]
    mimo_64_results = [row for row in results if row[0] == 64 and row[1] == 1 and row[2] == 16 and row[3] == True and row[4] == False]
    mimo_200_results = [row for row in results if row[0] == 200 and row[1] == 1 and row[2] == 16 and row[3] == True and row[4] == False]

    # filter out D1 != 1 results
    max_200_results = [row for row in max_200_results if row[10] == 1]
    max_4000_results = [row for row in max_4000_results if row[10] == 1]
    tsp_200_results = [row for row in tsp_200_results if row[10] == 1]
    tsp_8000_results = [row for row in tsp_8000_results if row[10] == 1]
    sudoku_200_results = [row for row in sudoku_200_results if row[10] == 1]
    sudoku_729_results = [row for row in sudoku_729_results if row[10] == 1]
    mimo_64_results = [row for row in mimo_64_results if row[10] == 1]
    mimo_200_results = [row for row in mimo_200_results if row[10] == 1]

    result_collect = [max_200_results, max_4000_results, tsp_200_results, tsp_8000_results, sudoku_200_results, sudoku_729_results, mimo_64_results, mimo_200_results]
    label_list = ["MaxCut-200", "MaxCut-4000", "TSP-200", "TSP-8000", "Sudoku-200", "Sudoku-729", "MIMO-64", "MIMO-200"]
    label_in_list = ["SACHI (baseline)", "+Opt. encoding", "+Opt. encoding + Opt. parallelism", "+Opt. encoding + Opt. parallelism + Opt. SRAM"]
    benchmark_name_in_list = ["200", "4000", "200", "8000", "200", "729", "64", "200"]
    component_list = ["mac", "spin_updating", "sram", "dram"]
    component_tag_list = ["MAC", "L1", "L2", "DRAM"]
    cycles_in_list = [[],[],[], []]
    energy_in_list = [[],[],[], []]
    cycle_breakdown_in_list = [[],[],[], []]
    energy_breakdown_in_list = [[],[],[], []]
    tops_in_list = [[],[],[], []]
    topsw_in_list = [[],[],[], []]
    topsmm2_in_list = [[],[],[], []]
    sorting_dict = {'topsmm2': 17, 'topsw': 16, 'cycles': 11, 'energy': 12, 'tops': 15}
    sorting_idx = sorting_dict['topsw']
    for result_set_idx in range(len(result_collect)):
        result_set = result_collect[result_set_idx]
        assert len(result_set) > 0, f"No results found for set index {result_set_idx}"
        result_set.sort(key=lambda x: -x[sorting_idx])
        row_best = result_set[0]
        logging.info(f"---{label_list[result_set_idx]}-------------------")
        row_baseline = [row for row in result_set if row[5] == "neighbor" and row[6] == 160 and row[7] == 80 and row[8] == 16 and row[9] == 10 and row[10] == 1][0]
        row_encoding_list = [row for row in result_set if row[6] == 160 and row[7] == 80 and row[8] == 16 and row[9] == 10 and row[10] == 1]
        row_encoding_list.sort(key=lambda x: -x[sorting_idx])
        row_encoding = row_encoding_list[0]
        row_encoding_mac_list = [row for row in result_set if row[6] == 160 and row[7] == 80]
        row_encoding_mac_list.sort(key=lambda x: -x[sorting_idx])
        row_encoding_mac = row_encoding_mac_list[0]
        logging.info(f"Baseline --> Cycles: {row_baseline[11]}, Energy: {row_baseline[12]}, TOPSMM2: {row_baseline[17]}, TOPSW: {row_baseline[16]}")
        logging.info(f"+Encode: {row_encoding[5]} --> Cycles: {row_encoding[11]}, Energy: {row_encoding[12]}, TOPSMM2: {row_encoding[17]}, TOPSW: {row_encoding[16]}")
        logging.info(f"++MAC: {row_encoding_mac[8]*row_encoding_mac[9]*row_encoding_mac[10]} --> Cycles: {row_encoding_mac[11]}, Energy: {row_encoding_mac[12]}, TOPSMM2: {row_encoding_mac[17]}, TOPSW: {row_encoding_mac[16]}")
        logging.info(f"+++SRAM: {row_best[6]}KB, CIM: depth: {row_best[7]}, size: {row_best[7] * row_best[9] * row_best[-1]['bit_per_weight']/8/1024}KB --> Cycles: {row_best[11]}, Energy: {row_best[12]}, TOPSMM2: {row_best[17]}, TOPSW: {row_best[16]}")

        # logging.info(f"Baseline --> parfor_hw: {row_baseline[-1]['parfor_hw']}, temfor_hw: {row_baseline[-1]['temfor_hw']}")
        # logging.info(f"+Encode: --> parfor_hw: {row_encoding[-1]['parfor_hw']}, temfor_hw: {row_encoding[-1]['temfor_hw']}")
        # logging.info(f"++MAC: --> parfor_hw: {row_encoding_mac[-1]['parfor_hw']}, temfor_hw: {row_encoding_mac[-1]['temfor_hw']}")
        # logging.info(f"+++SRAM: --> parfor_hw: {row_best[-1]['parfor_hw']}, temfor_hw: {row_best[-1]['temfor_hw']}")

        # collect data for plotting
        cycles_in_list[0].append(row_baseline[11])
        cycles_in_list[1].append(row_encoding[11])
        cycles_in_list[2].append(row_encoding_mac[11])
        cycles_in_list[3].append(row_best[11])
        energy_in_list[0].append(row_baseline[12])
        energy_in_list[1].append(row_encoding[12])
        energy_in_list[2].append(row_encoding_mac[12])
        energy_in_list[3].append(row_best[12])
        cycle_breakdown_in_list[0].append(row_baseline[-1]['latency_breakdown_plot'])
        cycle_breakdown_in_list[1].append(row_encoding[-1]['latency_breakdown_plot'])
        cycle_breakdown_in_list[2].append(row_encoding_mac[-1]['latency_breakdown_plot'])
        cycle_breakdown_in_list[3].append(row_best[-1]['latency_breakdown_plot'])
        energy_breakdown_in_list[0].append(row_baseline[-1]['energy_breakdown_plot'])
        energy_breakdown_in_list[1].append(row_encoding[-1]['energy_breakdown_plot'])
        energy_breakdown_in_list[2].append(row_encoding_mac[-1]['energy_breakdown_plot'])
        energy_breakdown_in_list[3].append(row_best[-1]['energy_breakdown_plot'])
        topsmm2_in_list[0].append(row_baseline[17])
        topsmm2_in_list[1].append(row_encoding[17])
        topsmm2_in_list[2].append(row_encoding_mac[17])
        topsmm2_in_list[3].append(row_best[17])
        topsw_in_list[0].append(row_baseline[16])
        topsw_in_list[1].append(row_encoding[16])
        topsw_in_list[2].append(row_encoding_mac[16])
        topsw_in_list[3].append(row_best[16])
        tops_in_list[0].append(row_baseline[15])
        tops_in_list[1].append(row_encoding[15])
        tops_in_list[2].append(row_encoding_mac[15])
        tops_in_list[3].append(row_best[15])

    plot_results_breakdown_in_bar_chart(
        cycles_in_list=cycles_in_list,
        cycles_breakdown_in_list=cycle_breakdown_in_list,
        energy_in_list=energy_in_list,
        energy_breakdown_in_list=energy_breakdown_in_list,
        label_in_list=label_in_list,
        benchmark_name_in_list=benchmark_name_in_list,
        title=None,
        component_list=component_list,
        component_tag_list=component_tag_list,
        topsmm2_in_list=topsmm2_in_list,
        topsw_in_list=topsw_in_list,
        log_scale=True
    )
    breakpoint()

    # pb_size = int(row[0])
    # degree_density = float(row[1])
    # weight_shared_precision = int(row[2])
    # with_bias = row[3] == 'True'
    # problem_specific_weight = row[4] == 'True'
    # encoding = row[5]
    # sram_size_in_KB = int(row[6])
    # cim_depth = int(row[7])
    # num_macros = int(row[8])
    # d2 = int(row[9])
    # d1 = int(row[10])
    # cycles_to_solution = int(row[11])
    # energy_to_solution = float(row[12])
    # tops = float(row[15])
    # topsw = float(row[16])
    # topsmm2 = float(row[17])
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_file_path = "outputs/expr_sweep_results.pkl"
    analysis_output_path = "outputs/expr_sweep_analysis.yaml"
    analyze_expr_sweep_results(results_file_path, analysis_output_path)