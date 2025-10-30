import itertools
import os
import logging
import matplotlib.pyplot as plt
import copy
import math
import numpy as np

def benchmark_library(name: str = ""):
    """ benchmark used in the paper """
    benchmarks: dict = {
        "g22_2K_IM": {"num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 36,
                      "latency": 1, "energy": 0, "latency_model": 0, "energy_model": 0},
        "g22_2K_BM": {"num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 36,
                      "latency": 0.4, "energy": 12.36, "latency_model": 0, "energy_model": 0},
        "g22_2K_SM": {"num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 36,
                      "latency": 1.1, "energy": 24.43, "latency_model": 0, "energy_model": 0},
        "g39_2K_IM": {"num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 209,
                      "latency": 1, "energy": 0, "latency_model": 0, "energy_model": 0},
        "g39_2K_BM": {"num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 209,
                      "latency": 0.6, "energy": 0, "latency_model": 0, "energy_model": 0},
        "g39_2K_SM": {"num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 209,
                      "latency": 1.05, "energy": 0, "latency_model": 0, "energy_model": 0},
        "K2000_2K_IM": {"num_spins": 2000, "num_js": 2000 * 2000, "num_iterations": 1, "w_pres": 2, "packet_pres": 2, "max_degree": 2000,
                        "latency": 1, "energy": 23.03, "latency_model": 0, "energy_model": 0},
    }
    return benchmarks[name]

def PRIM_CAEFA_hw_model(
        # Hardware of PRIM_CAEFA
        PRIM_CAEFA_hw: dict = {"sram_size_Mb": 8, "sram_area": 2.615427, "compute_size_Mb": 8, "compute_area": 2.55 * 2.59 - 2.615427,
                                    "sram_area_per_Mbit": 2.615427 / 8, "compute_area_per_Mb": (2.55 * 2.59 - 2.615427) / 8},  # area: mm2
        PRIM_CAEFA_power: dict = {"IM_power_per_degree_per_bit": 0.5757 / 2 * 1e-8,
                                  "BM_power_per_degree_per_bit": 0.3091 / 2 * 1e-6,
                                  "SM_power_per_degree_per_bit": 0.6110 / 4 * 1e-6}, # /2, /2, /4 are the packet_pres of the corresponding mode
        sram_setting: dict = {"size_Mb": 8, "bw": 1024},  # bw: bit width; area: mm2; @TSMC40nm

        # benchmark
        benchmark_dict: dict = {"name": "IM", "file_path": '../problems/g22.rud', "num_iterations": 1, "w_pres": 1, "packet_pres": 2},
        tclk: int = 1,
):

    # get the info of testbench
    data = np.loadtxt(benchmark_dict["file_path"])
    num_spins = int(data[0][0])
    num_spins_bw = np.ceil(np.log2(num_spins))
    num_js = int(data[0][1] * 2)
    degree_per_spin = list(range(num_spins))
    sj_ID = [[] for _ in range(num_spins)]
    for row in data[1:]:
        a, b = int(row[0]-1), int(row[1]-1)
        sj_ID[a].append(b)
        sj_ID[b].append(a)
    sj_ID_sorted = [sorted(row) for row in sj_ID]
    latency_per_spin = {"latency_read_si": list(range(num_spins)),
                        "latency_read_sj": list(range(num_spins)),
                        "latency_read_Jij": list(range(num_spins))}
    for i in range(num_spins):
        degree_per_spin[i] = int(np.sum(data[:, :2] == i+1))

    # calculate the total area
    area_collect = {"sram": sram_setting["size_Mb"] * PRIM_CAEFA_hw["sram_area_per_Mbit"],
                    "compute": sram_setting["size_Mb"] * PRIM_CAEFA_hw["compute_area_per_Mb"]}
    area_total = sum([area_collect[key] for key in area_collect.keys()])

    # calculate the latency
    latency_collect = {"sram": 1, "spin update": num_spins}
    if 'IM' in benchmark_dict["name"]:
        latency_collect["sram"] = int((np.ceil(num_spins * benchmark_dict["packet_pres"] / sram_setting["bw"]))) * num_spins
    elif 'BM' in benchmark_dict["name"]:
        first_packet_bw = num_spins_bw * 2 + benchmark_dict["packet_pres"]
        other_packet_bw = num_spins_bw + benchmark_dict["packet_pres"]
        num_packets_first_word = np.floor((sram_setting["bw"] - first_packet_bw) / other_packet_bw) + 1
        num_packets_other_word = np.floor(sram_setting["bw"] / other_packet_bw)
        for i in range(num_spins):
            if degree_per_spin[i] <= num_packets_first_word:
                latency_per_spin["latency_read_Jij"][i] = 1
            else:
                latency_per_spin["latency_read_Jij"][i] = math.ceil((degree_per_spin[i] - num_packets_first_word) / num_packets_other_word) + 1
        if max(latency_per_spin["latency_read_Jij"]) > 2:
            latency_per_spin["latency_read_Jij"] = [2 if degree == 1 else degree for degree in latency_per_spin["latency_read_Jij"]]
        latency_per_spin["latency_read_si"] = [0]
        latency_per_spin["latency_read_sj"] = [0]
        latency_collect["sram"] = sum(latency_per_spin["latency_read_Jij"])
    elif 'SM' in benchmark_dict["name"]:
        first_packet_bw = num_spins_bw * 2 + 1 + benchmark_dict["packet_pres"]
        other_packet_bw = num_spins_bw + benchmark_dict["packet_pres"]
        num_packets_first_word = np.floor((sram_setting["bw"] - first_packet_bw) / other_packet_bw) + 1
        num_packets_other_word = np.floor(sram_setting["bw"] / other_packet_bw)
        remain_packets = num_packets_first_word
        remain_bits = sram_setting["bw"]
        latency_per_spin["latency_read_si"] = [num_spins]
        remain_cache_flag_en = 0
        for i in range(num_spins):
            if degree_per_spin[i] <= remain_packets:
                latency_per_spin["latency_read_Jij"][i] = 1
                latency_per_spin["latency_read_sj"][i] = len(set([math.floor(x/sram_setting["bw"]) for x in sj_ID_sorted[i]]))
                if remain_cache_flag_en == 1:
                    remain_bits = remain_bits - degree_per_spin[i] * other_packet_bw
                else:
                    remain_bits = remain_bits - degree_per_spin[i] * other_packet_bw - (first_packet_bw-other_packet_bw)
            else:
                num_Jij_per_cycle = [remain_packets]
                latency_per_spin["latency_read_Jij"][i] = math.ceil((degree_per_spin[i] - remain_packets) / num_packets_other_word) + 1
                num_Jij_last_cycle = (degree_per_spin[i] - remain_packets) % num_packets_other_word
                remain_bits = sram_setting["bw"] - num_Jij_last_cycle * other_packet_bw
                for j in range(1,latency_per_spin["latency_read_Jij"][i]):
                    num_Jij_per_cycle.append(num_packets_other_word)
                num_Jij_per_cycle.append(num_Jij_last_cycle)
                num_Jij_sum = 0
                latency_per_spin["latency_read_sj"][i] = 0
                for j in range(latency_per_spin["latency_read_Jij"][i]):
                    cur_num_Jij_sum = int(num_Jij_sum + num_Jij_per_cycle[j])
                    latency_per_spin["latency_read_sj"][i] = latency_per_spin["latency_read_sj"][i] + \
                            len(set([math.floor(x/sram_setting["bw"]) for x in sj_ID_sorted[i][num_Jij_sum:cur_num_Jij_sum]]))
                    num_Jij_sum = cur_num_Jij_sum

            if remain_bits < (num_spins_bw + 1):
                remain_bits = sram_setting["bw"]
                remain_packets = math.floor((remain_bits - first_packet_bw) / other_packet_bw) + 1
                remain_cache_flag_en = 0
            elif remain_bits < first_packet_bw:
                remain_bits = sram_setting["bw"]
                remain_packets = math.floor(remain_bits / other_packet_bw)
                remain_cache_flag_en = 1
            else:
                remain_packets = math.floor((remain_bits - first_packet_bw) / other_packet_bw) + 1
                remain_cache_flag_en = 0

        latency_collect["sram"] = sum([num for sublist in latency_per_spin.values() for num in sublist])

    # calculate the energy
    if 'IM' in benchmark_dict["name"]:
        power = PRIM_CAEFA_power["IM_power_per_degree_per_bit"] \
                      * benchmark_dict["packet_pres"] * num_spins * num_spins * benchmark_dict["num_iterations"]
    elif 'BM' in benchmark_dict["name"]:
        power = PRIM_CAEFA_power["BM_power_per_degree_per_bit"] \
                    * benchmark_dict["packet_pres"] * num_js * benchmark_dict["num_iterations"]
    elif 'SM' in benchmark_dict["name"]:
        power = PRIM_CAEFA_power["SM_power_per_degree_per_bit"] * benchmark_dict["packet_pres"] * num_js * benchmark_dict["num_iterations"]

    # calculate the overall system latency and energy
    latency_system = sum(latency_collect.values())
    energy_collect = {"all": power * latency_system * tclk}
    energy_system = sum(energy_collect.values())

    # print the results
    logging.info(f"Benchmark: {benchmark_dict}")
    logging.info(f"System Latency [cycles]: {latency_system}, Latency [ns]: {latency_system * tclk}, Energy [nJ]: {energy_system/1000}")
    logging.info(f"Area [mm2]: {area_collect}, Total Area [mm2]: {area_total}")
    # return the performance: latency breakdown, energy breakdown, area breakdown, latency in cycles, latency in ns, energy in pj, area in mm2
    return latency_collect, energy_collect, area_collect, latency_system, latency_system * tclk, energy_system, area_total


def plot_results_in_pie_chart(benchmark_name: str, latency_collect: dict, energy_collect: dict, area_collect: dict, output_file: str = "output/PRIM_CAEFA.png"):
    """ plot the latency (left) in bar, and energy breakdown (right) in pie chart """
    # calculate the total latency, energy, area
    total_latency = sum(latency_collect.values())
    total_energy = sum(energy_collect.values())
    total_area = sum(area_collect.values())
    # plotting the results
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))

    """ plotting the latency breakdown """
    labels = list(latency_collect.keys())
    sizes = list(latency_collect.values())
    ax[0].pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%', startangle=90, wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold'})
    ax[0].axis('equal')
    ax[0].set_title(f'Latency [{round(total_latency, 2)} cycles]@{benchmark_name}', weight='bold')

    """ plotting the energy breakdown """
    labels = list(energy_collect.keys())
    sizes = [energy_collect["all"]]
    ax[1].pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%', startangle=90, wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold'})
    ax[1].axis('equal')
    ax[1].set_title(f'Energy [{round(total_energy, 2)} nJ]@{benchmark_name}', weight='bold')

    """ plotting the area breakdown """
    labels = list(area_collect.keys())
    sizes = list(area_collect.values())
    ax[2].pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%', startangle=90, wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold'})
    ax[2].axis('equal')
    ax[2].set_title(f'Area [{round(total_area, 2)} mm2]@{benchmark_name}',
                    weight='bold')
    # save the figure
    plt.tight_layout()
    plt.savefig(output_file)

def plot_results_in_bar(benchmark_list, latency_cycles_results, energy_breakdown_results, throughput_results, energy_efficiency_results, area_efficiency_results, output_file="output/sachi.png"):
    """ plot the modeling results in bar chart """
    colors = [u'#cd87de', u'#fff6d5', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    fig, ax = plt.subplots(1, 5, figsize=(20, 6))
    benchmark_names = benchmark_list
    for i in range(5):
        ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[i].set_axisbelow(True)
        ax[i].set_yscale('log')
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha='right')
    # plot the latency bar chart (cycles, system)
    x = list(range(len(benchmark_names)))
    width = 0.35
    ax[0].bar(x, latency_cycles_results, width, color=colors[0], edgecolor='black')
    ax[0].set_ylabel('Latency [cycles]', weight='bold')
    ax[0].set_title('Latency (System)')
    ax[0].set_xticks([i for i in x])
    ax[0].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[0].text(i, latency_cycles_results[i] * 1.01, f"{latency_cycles_results[i]:.2e}", ha='center', va='bottom', weight='bold')

    # plot the energy breakdown bar chart (stacked)
    energy = list(itertools.chain(*[list(energy_breakdown_results[tb].values()) for tb in range(len(energy_breakdown_results))]))
    ax[1].bar(x, energy, width=width, color=colors[1], edgecolor='black')
    ax[1].set_ylabel('Energy [nJ]', weight='bold')
    ax[1].set_title('Energy Breakdown')
    ax[1].set_xticks([i for i in x])
    ax[1].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[1].text(i, energy[i] * 1.01, f"{energy[i]:.2e}", ha='center', va='bottom', weight='bold')

    # plot the throughput bar
    ax[2].bar(x, throughput_results, width, color=colors[2], edgecolor='black')
    ax[2].set_ylabel('Throughput [Iter/s]', weight='bold')
    ax[2].set_title('Throughput')
    ax[2].set_xticks([i for i in x])
    ax[2].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[2].text(i, throughput_results[i] * 1.01, f"{throughput_results[i]:.2e}", ha='center', va='bottom', weight='bold')

    # plot the energy efficiency bar
    ax[3].bar(x, energy_efficiency_results, width, color=colors[3], edgecolor='black')
    ax[3].set_ylabel('Energy Efficiency [Iter/J]', weight='bold')
    ax[3].set_title('Energy Efficiency')
    ax[3].set_xticks([i for i in x])
    ax[3].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[3].text(i, energy_efficiency_results[i] * 1.01, f"{energy_efficiency_results[i]:.2e}", ha='center', va='bottom', weight='bold')

    # plot the area efficiency bar
    ax[4].bar(x, area_efficiency_results, width, color=colors[4], edgecolor='black')
    ax[4].set_ylabel('Area Efficiency [Iter/s/mm2]', weight='bold')
    ax[4].set_title('Area Efficiency')
    ax[4].set_xticks([i for i in x])
    ax[4].set_xticklabels(benchmark_names)
    for i in range(len(benchmark_names)):
        ax[4].text(i, area_efficiency_results[i] * 1.01, f"{area_efficiency_results[i]:.2e}", ha='center', va='bottom', weight='bold')
    # save the figure
    plt.tight_layout()
    plt.savefig(output_file)

if __name__ == "__main__":
    """ modeling the hardware architecture performance of the PRIM-CAEFA hw """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    """ experiment settings """
    problem_list = [22, 39]
    operation_mode = ['IM', 'BM', 'SM']
    packet_pres_list = [2, 2, 4]
    node_count_list = [2000, 2000]
    w_precision = 1
    testcase = "g"
    """ experiment settings end """
    benchmark_name_list = [f"{testcase}{count}" for count in problem_list for mode in operation_mode]
    benchmark_name_mode_list = [f"{testcase}{count}_{mode}" for count in problem_list for mode in operation_mode]
    output_file = f"output/PRIM-CAEFA_{testcase}.png"
    """ experiment below """
    latency_cycles_results = []
    energy_breakdown_results = []
    throughput_results = []
    energy_efficiency_results = []
    area_efficiency_results = []
    for benchmark_idx in range(len(problem_list)):
        for mode in range(len(operation_mode)):
            benchmark_name = benchmark_name_list[benchmark_idx * len(operation_mode) + mode]
            benchmark_name_mode = benchmark_name_mode_list[benchmark_idx * len(operation_mode) + mode]
            logging.info(f"Modeling the performance of {benchmark_name} in operation mode {operation_mode[mode]}")
            # change the benchmark if the graph topology (e.g., TSP/King's graph) changes
            sram_setting: dict = {"size_Mb": 8, "bw": 1024},  # bw: bit width; area: mm2; @TSMC40nm
            benchmark = {"name": operation_mode[mode], "file_path": os.path.join("../problems/" + benchmark_name + ".rud"),
                         "num_iterations": 1, "w_pres": w_precision, "packet_pres": packet_pres_list[mode]}
            latency_collect, energy_collect, area_collect, latency_system_cc, latency_system_ns, energy_system, area_total = PRIM_CAEFA_hw_model(
                        benchmark_dict = benchmark,
                        tclk = 40,  # ns, suppose the targeted frequency is 25MHz
                        )
            iteration_per_sec = 1 / (latency_system_ns * 1e-9)
            iteration_per_joule = 1 / (energy_system * 1e-9)
            iteraton_per_sec_per_mm2 = iteration_per_sec / area_total
            logging.info(f"Iter/s: {iteration_per_sec}, Iter/s/W: {iteration_per_joule}, Iter/s/mm2: {iteraton_per_sec_per_mm2}")
            plot_results_in_pie_chart(benchmark_name_mode, latency_collect, energy_collect, area_collect, output_file=f"output/PRIM_CAEFA_{benchmark_name_mode}.png")
            latency_cycles_results.append(latency_system_cc)
            energy_breakdown_results.append(energy_collect)
            throughput_results.append(iteration_per_sec)
            energy_efficiency_results.append(iteration_per_joule)
            area_efficiency_results.append(iteraton_per_sec_per_mm2)
    plot_results_in_bar(benchmark_name_mode_list, latency_cycles_results, energy_breakdown_results,
                    throughput_results, energy_efficiency_results, area_efficiency_results, output_file=output_file)