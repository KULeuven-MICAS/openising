import logging
import matplotlib.pyplot as plt
import copy
import math
import numpy as np


def benchmark_library(name: str = "MD"):
    """benchmark used in the paper"""
    benchmarks: dict = {
        "MD_500": {  # King graph
            "name": "MD_500",
            "num_spins": 500,
            "num_degrees": 8,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
        "MD_100K": {  # King graph
            "name": "MD_100K",
            "num_spins": 100 * 1000,
            "num_degrees": 8,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
        "MD_200K": {  # King graph
            "name": "MD_200K",
            "num_spins": 200 * 1000,
            "num_degrees": 8,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
        "MD_300K": {  # King graph
            "name": "MD_300K",
            "num_spins": 300 * 1000,
            "num_degrees": 8,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
        "MD_1M": {  # King graph
            "name": "MD_1M",
            "num_spins": 1000 * 1000,
            "num_degrees": 8,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
        "TSP_1K": {  # complete graph
            "name": "TSP_1K",
            "num_spins": 1000,
            "num_degrees": 999,
            "w_pres": 2,
            "num_iterations": 1,
            "num_trails": 1,
        },
    }
    return benchmarks[name]


def sachi_hw_model(
    dram_setting: dict = {
        "size_gb": 800,
        "bw": 128,
        "pj_per_bit": 3.7,
        "double_buf": True,
    },  # bw: bit width @ DDR5
    sram_setting: dict = {
        "size_kb": 160,
        "bw": 1024,
        "pj_w_per_bit": 0.04987,
        "pj_r_per_bit": 0.0587,
        "area": 0.04975,
        "double_buf": True,
    },  # bw: bit width; area: mm2; @TSMC28nm
    compute_memory_setting: dict = {
        "depth": 80,
        "bw": 1024,
        "pj_w_per_bit": 0.0157,
        "pj_r_per_bit": 0.0089,
        "area": 0.03636,
        "double_buf": False,
    },  # bw: bit width; area: mm2; @TSMC28nm
    num_cores_hw: int = 16,
    pj_compute_per_bit: float = 1.25
    * ((28 / 45) ** 2)
    * ((0.9) ** 2),  # pJ/bit@@TSMC28nm, Vdd=0.9V
    mm2_compute_per_bit: float = 0.614 * 1e-6,  # mm2/bit
    tclk: int = 1,  # ns
    benchmark_dict: dict = {
        "name": "Molecular Dynamics",
        "num_spins": 500,
        "num_degrees": 8,
        "num_iterations": 1,
        "w_pres": 2,
        "num_trails": 1,
    },
    mapping: dict = {"d1": "j", "d2": "l", "d3": "j"},
):
    """
    The entire sachi hw architecture is shown below:
                bw_dram              sram_bw
    | DRAM |   ------->   | SRAM |   ------>   | Compute Memory |   ------>   | Compute Logics |

    For loop dimension definitions: i+1[t][j] += i[t][j] * w[j][l]
    - t: trail
    - i: iteration
    - j: spin
    - l: degree
    """
    """ The energy of updating adjacent matrixs IS considered in this model """
    """ There is no energy of annealing (random flipping, threshold comparison), as SACHI uses HNN """
    # scale the num_cores, as the sachi needs to store each spin twice
    num_cores = num_cores_hw // 2
    # calculate the total area
    area_collect = {
        "sram": sram_setting["area"],
        "compute_memory": compute_memory_setting["area"] * num_cores_hw,
        "compute": mm2_compute_per_bit * compute_memory_setting["bw"] * num_cores_hw,
    }
    area_total = sum([area_collect[key] for key in area_collect.keys()])
    # derive the hardware parallelism
    spin_pres = 1  # spin precision (0 or 1)
    parallelism: dict = {
        "d1": 1,
        "d2": int(
            compute_memory_setting["bw"] / (spin_pres + benchmark_dict["w_pres"])
        ),
        "d3": num_cores,
    }
    # derive the workload representation
    workload: dict = {
        "t": benchmark_dict["num_trails"],
        "i": benchmark_dict["num_iterations"],
        "j": benchmark_dict["num_spins"],
        "l": benchmark_dict["num_degrees"],
    }
    # derive the mapping: spatial
    parfor_hw: dict = {"d1": 1, "d2": 1, "d3": 1}
    parfor_sw: list = [(key, 1) for key in workload.keys()]
    for d, item in mapping.items():
        if (
            d == "d3"
            and workload[item] / compute_memory_setting["depth"] < parallelism[d]
        ):
            # special case for sachi: the d3 parallelism is limited by the compute memory depth
            parfor_hw[d] = workload[item] / compute_memory_setting["depth"]
        else:
            parfor_hw[d] = min(parallelism[d], workload[item])
        for i, (key, value) in enumerate(parfor_sw):
            if key == mapping[d]:
                parfor_sw[i] = (key, value * parfor_hw[d])
    # calculate the left workload loop size
    left_workload: dict = copy.deepcopy(workload)
    for d, item in mapping.items():
        left_workload[item] /= parfor_hw[d]
    for d, item in mapping.items():
        left_workload[item] = math.ceil(left_workload[item])
    # derive the mapping: temporal
    temfor_sw: list = [
        ("t", left_workload["t"]),
        ("i", left_workload["i"]),
        ("j", left_workload["j"]),
        ("l", left_workload["l"]),
    ]
    # allocate the temporal loops to the memories
    mem_sizes_bit: dict = {
        "dram": dram_setting["size_gb"] * 1024**3 * 8,
        "sram": sram_setting["size_kb"] * 1024 * 8,
        "compute_memory": compute_memory_setting["depth"]
        * compute_memory_setting["bw"]
        * num_cores,
    }
    temfor_hw: dict = {"dram": [], "sram": [], "compute_memory": []}
    unallocated_loops = copy.deepcopy(
        temfor_sw[2:]
    )  # neglect the trail and iteration loops
    allocated_loops_total = []
    for mem in [
        "compute_memory",
        "sram",
        "dram",
    ]:  # from the lowest level to the highest level
        if unallocated_loops == []:
            break
        allocated_loops = []
        if mem == "compute_memory":
            allowed_loop_size = compute_memory_setting["depth"]
        else:
            # calculate the minimal required mem size
            mem_sizes_bit_min = (
                (spin_pres + benchmark_dict["w_pres"])
                * math.prod(parfor_hw.values())
                * math.prod([value for key, value in allocated_loops_total])
            )  # for weight
            allowed_loop_size = mem_sizes_bit[mem] / mem_sizes_bit_min
        if allowed_loop_size > 1:
            for idx in range(len(unallocated_loops) - 1, -1, -1):
                if unallocated_loops[idx][1] <= allowed_loop_size:
                    allocated_loops = [unallocated_loops[idx]] + allocated_loops
                    allowed_loop_size /= unallocated_loops[idx][1]
                    if idx == 0:  # all the loops are allocated
                        unallocated_loops = []
                else:
                    allocated_loops = [
                        (unallocated_loops[idx][0], int(allowed_loop_size))
                    ] + allocated_loops
                    unallocated_loops[idx] = (
                        unallocated_loops[idx][0],
                        math.ceil(unallocated_loops[idx][1] / int(allowed_loop_size)),
                    )
                    unallocated_loops = unallocated_loops[: idx + 1]
                    break
        temfor_hw[mem] = allocated_loops
        allocated_loops_total = allocated_loops + allocated_loops_total
    # calculate the top-level memory idx
    mem_list = ["compute_memory", "sram", "dram"]
    mem_bw_list = [compute_memory_setting["bw"], sram_setting["bw"], dram_setting["bw"]]
    mem_dbuf_list = [
        compute_memory_setting["double_buf"],
        sram_setting["double_buf"],
        dram_setting["double_buf"],
    ]
    top_mem_idx = len(mem_list) - 1
    for mem_idx in range(len(mem_list)):
        if temfor_hw[mem_list[mem_idx]] == []:
            top_mem_idx = mem_idx - 1
            break
    # calculate the latency
    # there are two steps in each iteration: computation and spin updating
    # (i.e., packet updating/adjacent matrix updating)
    # latency of step one: comptutation
    ideal_latency = math.prod([value for key, value in temfor_sw])
    latency_collect: dict = {"compute": ideal_latency}
    access_collect: dict = {}
    parfor_size = math.prod([value for key, value in parfor_sw])
    for mem_idx in range(top_mem_idx + 1):
        # tmfor loops below the current mem level
        tmfor_size_lower = math.prod(
            [value for mem in mem_list[:mem_idx] for key, value in temfor_hw[mem]]
        )
        tmfor_size_total = math.prod(
            [value for key, value in temfor_sw]
        )  # consider t and i
        tmfor_size_upper = (
            tmfor_size_total / tmfor_size_lower
        )  # tmfor loops below the current mem level
        # calculate the write latency of tmfor_size_lower: wr_from_high
        if mem_idx == top_mem_idx:
            cycles_wr_per_tile = 0
        elif mem_idx == 0:  # compute_memory
            cycles_wr_per_tile = tmfor_size_lower * 2  # 2: sachi stores the spins twice
        else:
            cycles_wr_per_tile = (
                tmfor_size_lower * parfor_size * (spin_pres + benchmark_dict["w_pres"])
            ) / (mem_bw_list[mem_idx])
            cycles_wr_per_tile = math.ceil(cycles_wr_per_tile)
        cycles_wr_from_high = math.ceil(cycles_wr_per_tile * tmfor_size_upper)
        # calculate the read latency of tmfor_size_lower: rd_to_low
        if mem_idx == 0:
            cycles_rd_per_tile = tmfor_size_lower
        else:
            cycles_rd_per_tile = (
                tmfor_size_lower * parfor_size * (spin_pres + benchmark_dict["w_pres"])
            ) / (mem_bw_list[mem_idx])
            cycles_rd_per_tile = math.ceil(cycles_rd_per_tile)
        cycles_rd_to_low = math.ceil(cycles_rd_per_tile * tmfor_size_upper)
        # calculate the write latency of tmfor_size_lower: wr_from_low
        if mem_idx == 0:
            cycles_wr_per_tile = 0  # no write back within one iteration
        else:
            cycles_wr_per_tile = (
                tmfor_size_lower * parfor_size * (spin_pres + benchmark_dict["w_pres"])
            ) / (mem_bw_list[mem_idx])
            cycles_wr_per_tile = math.ceil(cycles_wr_per_tile)
        cycles_wr_from_low = math.ceil(cycles_wr_per_tile * tmfor_size_upper)
        # calculate the read latency of tmfor_size_lower: rd_to_high
        if mem_idx == top_mem_idx:
            cycles_rd_per_tile = 0
        elif mem_idx == 0:
            cycles_rd_per_tile = 0
        else:
            cycles_rd_per_tile = (
                tmfor_size_lower * parfor_size * (spin_pres + benchmark_dict["w_pres"])
            ) / (mem_bw_list[mem_idx])
            cycles_rd_per_tile = math.ceil(cycles_rd_per_tile)
        cycles_rd_to_high = math.ceil(cycles_rd_per_tile * tmfor_size_upper)
        # calculate the latency, considering double buffer
        if mem_dbuf_list[mem_idx]:
            latency_collect[mem_list[mem_idx]] = max(
                cycles_wr_from_high, cycles_rd_to_low
            ) + max(cycles_wr_from_low, cycles_rd_to_high)
        else:
            latency_collect[mem_list[mem_idx]] = (
                cycles_wr_from_high
                + cycles_rd_to_low
                + cycles_wr_from_low
                + cycles_rd_to_high
            )
        # calculate the access count
        access_collect[mem_list[mem_idx]] = {
            "wr": cycles_wr_from_high + cycles_wr_from_low,
            "rd": cycles_rd_to_low + cycles_rd_to_high,
        }
    # latency of step two: spin updating
    # in this step, all the data must be read out from the top memory and write back to the top memory.
    data_size_bit = (
        workload["j"] * workload["l"] * (spin_pres + benchmark_dict["w_pres"])
    )
    for mem_idx in [top_mem_idx]:
        if (
            mem_idx == 0
        ):  # the data within the compute memory may not be densely stored (spatial utilization on D2 < 1)
            cycles_rd_spin_updating = latency_collect[mem_list[mem_idx]]
            cycles_wr_spin_updating = latency_collect[mem_list[mem_idx]]
        else:
            cycles_rd_spin_updating = math.ceil(data_size_bit / mem_bw_list[mem_idx])
            cycles_wr_spin_updating = math.ceil(data_size_bit / mem_bw_list[mem_idx])
        access_spin_updating: dict = {
            "wr": cycles_wr_spin_updating * 2,
            "rd": cycles_rd_spin_updating * 2,
        }  # 2: sachi stores spins twice
        if mem_dbuf_list[mem_idx]:
            cycles_spin_updating = max(cycles_rd_spin_updating, cycles_wr_spin_updating)
        else:
            cycles_spin_updating = cycles_rd_spin_updating + cycles_wr_spin_updating
    # calculate the energy of step one: computation
    compute_energy = (
        pj_compute_per_bit * parfor_size * benchmark_dict["w_pres"] * ideal_latency
    )
    energy_collect: dict = {"compute": compute_energy}
    mem_energy_wr_per_bit = {
        "dram": dram_setting["pj_per_bit"],
        "sram": sram_setting["pj_w_per_bit"],
        "compute_memory": compute_memory_setting["pj_w_per_bit"],
    }
    mem_energy_rd_per_bit = {
        "dram": dram_setting["pj_per_bit"],
        "sram": sram_setting["pj_r_per_bit"],
        "compute_memory": compute_memory_setting["pj_r_per_bit"],
    }
    for mem_idx in range(top_mem_idx + 1):
        mem = mem_list[mem_idx]
        energy_collect[mem] = {}
        for key in ["wr", "rd"]:
            if key == "wr":
                energy_collect[mem][key] = (
                    access_collect[mem][key]
                    * mem_bw_list[mem_idx]
                    * mem_energy_wr_per_bit[mem]
                )
                if mem_idx == 0:  # compute memory
                    energy_collect[mem][key] *= 2  # 2: sachi stores spins twice
            else:
                energy_collect[mem][key] = (
                    access_collect[mem][key]
                    * mem_bw_list[mem_idx]
                    * mem_energy_rd_per_bit[mem]
                )

    # calculate the energy of step two: spin updating
    for mem_idx in [top_mem_idx]:
        mem = mem_list[mem_idx]
        energy_collect[f"{mem}_SU"] = {}
        for key in ["wr", "rd"]:
            if key == "wr":
                energy_collect[f"{mem}_SU"][key] = (
                    access_spin_updating[key]
                    * mem_bw_list[mem_idx]
                    * mem_energy_wr_per_bit[mem]
                )
            else:
                energy_collect[f"{mem}_SU"][key] = (
                    access_spin_updating[key]
                    * mem_bw_list[mem_idx]
                    * mem_energy_rd_per_bit[mem]
                )
    # calculate the overall system latency and energy
    latency_system = max(list(latency_collect.values())) + cycles_spin_updating
    energy_system = (
        sum([sum(energy_collect[mem].values()) for mem in mem_list[: top_mem_idx + 1]])
        + compute_energy
    )
    # extra: for the ease of plotting results later, add cycles_spin_updating within the latency_collect
    latency_collect[f"{mem_list[mem_idx]}_SU"] = cycles_spin_updating
    # print the results
    logging.info(f"Benchmark: {benchmark_dict}")
    logging.info(f"Parfor HW: {parfor_hw}")
    logging.info(f"Parfor SW: {parfor_sw}")
    logging.info(f"Temfor HW: {temfor_hw}")
    logging.info(f"Temfor SW: {temfor_sw}")
    logging.info("---- Performance metrics including all iterations and trials ----")
    logging.info(f"Latency [cycles]: {latency_collect}")
    logging.info(f"Energy [pJ]: {energy_collect}")
    logging.info(
        f"Ideal Latency [cycles]: {ideal_latency}, Latency [ns]: {ideal_latency * tclk}, "
        f"Energy [nJ]: {compute_energy/1000}"
    )
    logging.info(
        f"System Latency [cycles]: {latency_system}, Latency [ns]: {latency_system * tclk}, "
        f"Energy [nJ]: {energy_system/1000}"
    )
    logging.info(f"Area [mm2]: {area_collect}, Total Area [mm2]: {area_total}")
    # return the performance: latency breakdown, energy breakdown, area breakdown,
    # latency in cycles, latency in ns, energy in pj, area in mm2
    return (
        latency_collect,
        energy_collect,
        area_collect,
        latency_system,
        latency_system * tclk,
        energy_system,
        area_total,
    )


def plot_results_in_pie_chart(
    benchmark_name: str,
    latency_collect: dict,
    energy_collect: dict,
    area_collect: dict,
    output_file: str = "output/sachi.png",
):
    """plot the latency (left) in bar, and energy breakdown (right) in pie chart"""
    # calculate the total latency, energy, area
    total_latency = sum(latency_collect.values())
    total_energy = energy_collect["compute"] + sum(
        [
            sum(energy_collect[mem].values())
            for mem in energy_collect.keys()
            if mem != "compute"
        ]
    )
    total_area = sum(area_collect.values())
    # plotting the results
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    labels = [key for key, value in latency_collect.items() if "SU" not in key]
    sizes = [value for key, value in latency_collect.items() if "SU" not in key]
    mem_su = [key for key, value in latency_collect.items() if "SU" in key][0].rstrip(
        "_SU"
    )
    sizes_su = []
    for mem_idx in range(len(labels)):
        if mem_idx == sizes.index(max(sizes)):
            latency_su = latency_collect[f"{mem_su}_SU"]
            sizes_su.append(latency_su)
        else:
            sizes_su.append(0)
    """ plotting the latency breakdown """
    ax[0].bar(labels, sizes, bottom=0, edgecolor="black", label="computation")
    ax[0].bar(
        labels, sizes_su, bottom=sizes, edgecolor="black", label="spin updating (SU)"
    )
    ax[0].set_ylabel("Latency [cycles]", weight="bold")
    ax[0].set_title(f"Latency [{total_latency} cycles]@{benchmark_name}", weight="bold")
    ax[0].legend()
    # add a star to the highest bar
    max_idx = sizes.index(max(sizes))
    ax[0].text(
        max_idx,
        sizes[max_idx] + latency_su,
        "\u2605",
        ha="center",
        va="center",
        weight="bold",
        fontsize=20,
    )
    # add the text labels to show the absolute values
    for i in range(len(labels)):
        if i == max_idx:
            ax[0].text(
                i,
                (sizes[i] + latency_su) * 1.01,
                f"{sizes[i]+latency_su}",
                ha="center",
                va="bottom",
                weight="bold",
            )
        else:
            ax[0].text(
                i,
                sizes[i] * 1.01,
                f"{sizes[i]}",
                ha="center",
                va="bottom",
                weight="bold",
            )
    # rotate the x-axis ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right")
    """ plotting the energy breakdown """
    labels = list(energy_collect.keys())
    sizes = [energy_collect["compute"]] + [
        sum(energy_collect[mem].values()) for mem in labels[1:]
    ]  # sum the energy of read and write, as the energy of read and write are the same
    ax[1].pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
        textprops={"weight": "bold"},
    )
    ax[1].axis("equal")
    ax[1].set_title(
        f"Energy [{round(total_energy, 2)} nJ]@{benchmark_name}", weight="bold"
    )
    """ plotting the area breakdown """
    labels = list(area_collect.keys())
    sizes = list(area_collect.values())
    ax[2].pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
        textprops={"weight": "bold"},
    )
    ax[2].axis("equal")
    ax[2].set_title(
        f"Area [{round(total_area, 2)} mm2]@{benchmark_name}", weight="bold"
    )
    # save the figure
    plt.tight_layout()
    plt.savefig(output_file)


def plot_results_in_bar(
    benchmark_list,
    latency_cycles_ideal_results,
    latency_cycles_results,
    energy_breakdown_results,
    throughput_results,
    energy_efficiency_results,
    area_efficiency_results,
    output_file="output/sachi.png",
):
    """plot the modeling results in bar chart"""
    colors = [
        "#cd87de",
        "#fff6d5",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    benchmark_names = benchmark_list
    for i in range(6):
        ax[i].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[i].set_axisbelow(True)
        if i != 2:
            ax[i].set_yscale("log")
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right")
    # plot the latency bar chart (cycles, compute)
    x = list(range(len(benchmark_names)))
    width = 0.35
    ax[0].bar(
        x, latency_cycles_ideal_results, width, color=colors[0], edgecolor="black"
    )
    ax[0].set_ylabel("Latency [cycles]", weight="bold")
    ax[0].set_title("Latency (Ideal)")
    ax[0].set_xticks([i for i in x])
    ax[0].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[0].text(
            i,
            latency_cycles_ideal_results[i] * 1.01,
            f"{latency_cycles_ideal_results[i]:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    # plot the latency bar chart (cycles, system)
    x = list(range(len(benchmark_names)))
    width = 0.35
    ax[1].bar(x, latency_cycles_results, width, color=colors[0], edgecolor="black")
    ax[1].set_ylabel("Latency [cycles]", weight="bold")
    ax[1].set_title("Latency (System)")
    ax[1].set_xticks([i for i in x])
    ax[1].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[1].text(
            i,
            latency_cycles_results[i] * 1.01,
            f"{latency_cycles_results[i]:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    # plot the energy breakdown bar chart (stacked)
    stacking_order = ["compute", "compute_memory", "sram", "dram"]
    # check the energy_breakdown_results, add dram: 0 if dram is not in the key
    for benchmark in energy_breakdown_results:
        for mem in stacking_order:
            if mem not in benchmark:
                benchmark[mem] = {"wr": 0, "rd": 0}
    x = list(range(len(benchmark_names)))
    width = 0.35
    bottom = np.zeros(len(benchmark_names))
    for mem_idx in range(len(stacking_order)):
        mem = stacking_order[mem_idx]
        if mem == "compute":
            energy = np.array(
                [energy_breakdown_results[x][mem] for x in range(len(benchmark_names))]
            )
        else:
            energy = np.array(
                [
                    sum(energy_breakdown_results[x][mem].values())
                    for x in range(len(benchmark_names))
                ]
            )
        ax[2].bar(
            x,
            energy,
            bottom=bottom,
            width=width,
            label=mem,
            color=colors[mem_idx],
            edgecolor="black",
        )
        bottom += energy
    ax[2].set_ylabel("Energy [nJ]", weight="bold")
    ax[2].set_title("Energy Breakdown")
    ax[2].legend()
    ax[2].set_xticks([i for i in x])
    ax[2].set_xticklabels(benchmark_names)
    # plot the throughput bar
    ax[3].bar(x, throughput_results, width, color=colors[2], edgecolor="black")
    ax[3].set_ylabel("Throughput [Iter/s]", weight="bold")
    ax[3].set_title("Throughput")
    ax[3].set_xticks([i for i in x])
    ax[3].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[3].text(
            i,
            throughput_results[i] * 1.01,
            f"{throughput_results[i]:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )

    # plot the energy efficiency bar
    ax[4].bar(x, energy_efficiency_results, width, color=colors[3], edgecolor="black")
    ax[4].set_ylabel("Energy Efficiency [Iter/J]", weight="bold")
    ax[4].set_title("Energy Efficiency")
    ax[4].set_xticks([i for i in x])
    ax[4].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[4].text(
            i,
            energy_efficiency_results[i] * 1.01,
            f"{energy_efficiency_results[i]:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )

    # plot the area efficiency bar
    ax[5].bar(x, area_efficiency_results, width, color=colors[4], edgecolor="black")
    ax[5].set_ylabel("Area Efficiency [Iter/s/mm2]", weight="bold")
    ax[5].set_title("Area Efficiency")
    ax[5].set_xticks([i for i in x])
    ax[5].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[5].text(
            i,
            area_efficiency_results[i] * 1.01,
            f"{area_efficiency_results[i]:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    # save the figure
    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    """modeling the hardware architecture performance of the sachi hw"""
    logging_level = logging.INFO  # logging level
    logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=logging_level, format=logging_format)
    """ experiment settings """
    # benchmark_name = "MD_1M"
    node_count_list = [500, 2000, 8000, 32000, 128000]
    w_precision = 4
    # node_count_list = [500, 100*1000, 200*1000, 300*1000, 1000*1000]  # this is the cases in the sachi paper
    testcase = "MD"
    """ experiment settings end """
    benchmark_list = [f"{testcase}_{count}" for count in node_count_list]
    output_file = f"output/sachi_{testcase}.png"
    sram_setting: dict = {
        "size_kb": 160,
        "bw": 1024,
        "pj_w_per_bit": 0.04987,
        "pj_r_per_bit": 0.0587,
        "area": 0.04975,
        "double_buf": True,
    }
    """ experiment below """
    latency_cycles_ideal_results = []
    latency_cycles_results = []
    energy_breakdown_results = []
    throughput_results = []
    energy_efficiency_results = []
    area_efficiency_results = []
    for benchmark_idx in range(len(benchmark_list)):
        benchmark_name = benchmark_list[benchmark_idx]
        logging.info(f"Modeling the performance of {benchmark_name}")
        # change the benchmark if the graph topology (e.g., TSP/King's graph) changes
        if testcase == "MD":
            benchmark: dict = {
                "name": benchmark_name,
                "num_spins": node_count_list[benchmark_idx],
                "num_degrees": 8,
                "w_pres": w_precision,
                "num_iterations": 1,
                "num_trails": 1,
            }
        else:  # TSP
            benchmark: dict = {
                "name": benchmark_name,
                "num_spins": node_count_list[benchmark_idx],
                "num_degrees": node_count_list[benchmark_idx],
                "w_pres": w_precision,
                "num_iterations": 1,
                "num_trails": 1,
            }
        (
            latency_collect,
            energy_collect,
            area_collect,
            latency_system_cc,
            latency_system_ns,
            energy_system,
            area_total,
        ) = sachi_hw_model(
            # from zigzag-imc model @ 28nm, VDD=0.9V
            # (assume 5 gates/bit, according to Fig. 13 in SACHI: 1 gate for XNOR, 2 for INV+1, 2 for 4:1 MUX)
            pj_compute_per_bit=0.567 * 5 * 1e-3,
            tclk=2,  # ns, suppose the targeted frequency is 500MHz
            benchmark_dict=benchmark,
            sram_setting=sram_setting,
        )
        iteration_per_sec = 1 / (latency_system_ns * 1e-9)
        iteration_per_joule = 1 / (energy_system * 1e-9)
        iteraton_per_sec_per_mm2 = iteration_per_sec / area_total
        logging.info(
            f"Iter/s: {iteration_per_sec}, Iter/s/W: {iteration_per_joule}, "
            f"Iter/s/mm2: {iteraton_per_sec_per_mm2}"
        )
        plot_results_in_pie_chart(
            benchmark_name,
            latency_collect,
            energy_collect,
            area_collect,
            output_file=f"output/sachi_{benchmark_name}.png",
        )
        latency_cycles_ideal_results.append(latency_collect["compute"])
        latency_cycles_results.append(latency_system_cc)
        energy_breakdown_results.append(energy_collect)
        throughput_results.append(iteration_per_sec)
        energy_efficiency_results.append(iteration_per_joule)
        area_efficiency_results.append(iteraton_per_sec_per_mm2)
    plot_results_in_bar(
        benchmark_list,
        latency_cycles_ideal_results,
        latency_cycles_results,
        energy_breakdown_results,
        throughput_results,
        energy_efficiency_results,
        area_efficiency_results,
        output_file=output_file,
    )
