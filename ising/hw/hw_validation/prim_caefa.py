import os.path
import sys
import logging
import numpy as np
import math
from api import plot_results_in_bar_chart_with_breakdown


def validation_to_prim_caefa():
    """
    validating the modeling results to PRIM_CAEFA (ASSCC'24)
    PRIM-CAEFA supports three operation modes: Interaction Mode (IM), Balance Mode (BM), Spin Mode (SM)
    validate to Fig 2 in PRIM-CAEFA, benchmark: G-set max-cut problem, size: 2k spins, w pres: 1bit
    The latency in PRIM-CAEFA is normalized to Interaction Mode
    the latency of SM is extracted from Fig.2
    """

    # HW settings
    IM_power_per_degree_per_bit = 0.5757 / 2 * 1e-8  # W/bit@40nm, Vdd=1.1V extracted from the paper
    BM_power_per_degree_per_bit = 0.3091 / 2 * 1e-6 # W/bit@40nm, Vdd=1.1V extracted from the paper
    SM_power_per_degree_per_bit = 0.6110 / 4 * 1e-6 # W/bit@40nm, Vdd=1.1V extracted from the paper
    SRAM_width = 1024
    # Benchmark settings
    benchmark_dict = {
        # latency [cycle]: reported latency per iteration, energy [nJ]: reported energy per iteration, latency_model [cycle]: latency to be modeled, energy_model [nJ]: energy to be modeled
        # num_js is not halved as PRIM-CAEFA stores J twice
        # w_pres is the precision of the problem, packet_pres is the precision used in the corresponding packet
        "G22_2K_IM": {"file_path": '../../benchmarks/G/G22.txt', "num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 2, "max_degree": 36,
                      "latency": 10000, "energy": 0, "latency_model": 0, "energy_model": 0},
        "G22_2K_BM": {"file_path": '../../benchmarks/G/G22.txt', "num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 2, "max_degree": 36,
                      "latency": 4000, "energy": 12.36 * 4000 * 40 / 1000, "latency_model": 0, "energy_model": 0},
        "G22_2K_SM": {"file_path": '../../benchmarks/G/G22.txt', "num_spins": 2000, "num_js": 19990 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 4, "max_degree": 36,
                      "latency": 11140, "energy": 24.43 * 11140 * 40 / 1000, "latency_model": 0, "energy_model": 0},
        "G39_2K_IM": {"file_path": '../../benchmarks/G/G39.txt', "num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 2, "max_degree": 209,
                      "latency": 10000, "energy": 0, "latency_model": 0, "energy_model": 0},
        "G39_2K_BM": {"file_path": '../../benchmarks/G/G39.txt', "num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 2, "max_degree": 209,
                      "latency": 6005, "energy": 0, "latency_model": 0, "energy_model": 0},
        "G39_2K_SM": {"file_path": '../../benchmarks/G/G39.txt', "num_spins": 2000, "num_js": 11778 * 2, "num_iterations": 1,
                      "w_pres": 2, "packet_pres": 4, "max_degree": 209,
                      "latency": 10363, "energy": 0, "latency_model": 0, "energy_model": 0},
        "K2000_2K_IM": {"file_path": '../../benchmarks/G/K2000.txt', "num_spins": 2000, "num_js": 2000 * 2000, "num_iterations": 1,
                        "w_pres": 2, "packet_pres": 2, "max_degree": 2000,
                        "latency": 10000, "energy": 23.03 * 10000 * 40 / 1000, "latency_model": 0, "energy_model": 0},
    }
    # nJ for the energy

    # hardware setting
    sram_setting: dict = {"size_Mb": 8, "bw": 1024}
    t_clk = 40 #ns

    # calculating the performance metrics
    for benchmark in benchmark_dict.keys():
        num_spins = benchmark_dict[benchmark]["num_spins"]
        num_js = benchmark_dict[benchmark]["num_js"]
        num_iterations = benchmark_dict[benchmark]["num_iterations"]
        w_pres = benchmark_dict[benchmark]["w_pres"]
        packet_pres = benchmark_dict[benchmark]["packet_pres"]
        max_degree = benchmark_dict[benchmark]["max_degree"]
        energy = benchmark_dict[benchmark]["energy"]
        latency = benchmark_dict[benchmark]["latency"]

        # get the info of testbench
        data = np.loadtxt(benchmark_dict[benchmark]["file_path"])
        num_spins = int(data[0][0])
        num_spins_bw = np.ceil(np.log2(num_spins))
        num_js = int(data[0][1] * 2)
        degree_per_spin = list(range(num_spins))
        sj_ID = [[] for _ in range(num_spins)]
        for row in data[1:]:
            a, b = int(row[0] - 1), int(row[1] - 1)
            sj_ID[a].append(b)
            sj_ID[b].append(a)
        sj_ID_sorted = [sorted(row) for row in sj_ID]
        latency_per_spin = {"latency_read_si": list(range(num_spins)),
                            "latency_read_sj": list(range(num_spins)),
                            "latency_read_Jij": list(range(num_spins))}
        for i in range(num_spins):
            degree_per_spin[i] = int(np.sum(data[:, :2] == i + 1))

        # degree_per_spin = np.ceil(num_js/num_spins)
        # IM_latency_model = 1

        # calculate the energy and latency
        latency_collect = {"sram": 1, "spin update": num_spins}
        if 'IM' in benchmark:
            power_model = IM_power_per_degree_per_bit * packet_pres * num_spins * num_spins * num_iterations
            latency_collect["sram"] = (np.ceil(
                num_spins * packet_pres / sram_setting["bw"])) * num_spins
            # latency_model = IM_latency_model
        elif 'BM' in benchmark:
            power_model = BM_power_per_degree_per_bit * packet_pres * num_js * num_iterations
            first_packet_bw = num_spins_bw * 2 + packet_pres
            other_packet_bw = num_spins_bw + packet_pres
            # latency_model = np.ceil(1 + (first_packet_bw + max_degree * other_packet_bw) / SRAM_width) \
            #                 / np.ceil(1 + num_spins * w_pres / SRAM_width)
            num_packets_first_word = np.floor((sram_setting["bw"] - first_packet_bw) / other_packet_bw) + 1
            num_packets_other_word = np.floor(sram_setting["bw"] / other_packet_bw)
            for i in range(num_spins):
                if degree_per_spin[i] <= num_packets_first_word:
                    latency_per_spin["latency_read_Jij"][i] = 1
                else:
                    latency_per_spin["latency_read_Jij"][i] = math.ceil(
                        (degree_per_spin[i] - num_packets_first_word) / num_packets_other_word) + 1
            if max(latency_per_spin["latency_read_Jij"]) > 2:
                latency_per_spin["latency_read_Jij"] = [2 if degree == 1 else degree for degree in
                                                        latency_per_spin["latency_read_Jij"]]
            latency_per_spin["latency_read_si"] = [0]
            latency_per_spin["latency_read_sj"] = [0]
            latency_collect["sram"] = sum(latency_per_spin["latency_read_Jij"])
        elif 'SM' in benchmark:
            power_model = SM_power_per_degree_per_bit * packet_pres * num_js * num_iterations
            first_packet_bw = num_spins_bw * 2 + 1 + packet_pres
            other_packet_bw = num_spins_bw + packet_pres
            num_packets_first_word = np.floor((sram_setting["bw"] - first_packet_bw) / other_packet_bw) + 1
            num_packets_other_word = np.floor(sram_setting["bw"] / other_packet_bw)
            remain_packets = num_packets_first_word
            remain_bits = sram_setting["bw"]
            latency_per_spin["latency_read_si"] = [num_spins]
            remain_cache_flag_en = 0
            for i in range(num_spins):
                if degree_per_spin[i] <= remain_packets:
                    latency_per_spin["latency_read_Jij"][i] = 1
                    latency_per_spin["latency_read_sj"][i] = len(
                        set([math.floor(x / sram_setting["bw"]) for x in sj_ID_sorted[i]]))
                    if remain_cache_flag_en == 1:
                        remain_bits = remain_bits - degree_per_spin[i] * other_packet_bw
                    else:
                        remain_bits = remain_bits - degree_per_spin[i] * other_packet_bw - (
                                    first_packet_bw - other_packet_bw)
                else:
                    num_Jij_per_cycle = [remain_packets]
                    latency_per_spin["latency_read_Jij"][i] = math.ceil(
                        (degree_per_spin[i] - remain_packets) / num_packets_other_word) + 1
                    num_Jij_last_cycle = (degree_per_spin[i] - remain_packets) % num_packets_other_word
                    remain_bits = sram_setting["bw"] - num_Jij_last_cycle * other_packet_bw
                    for j in range(1, latency_per_spin["latency_read_Jij"][i]):
                        num_Jij_per_cycle.append(num_packets_other_word)
                    num_Jij_per_cycle.append(num_Jij_last_cycle)
                    num_Jij_sum = 0
                    latency_per_spin["latency_read_sj"][i] = 0
                    for j in range(latency_per_spin["latency_read_Jij"][i]):
                        cur_num_Jij_sum = int(num_Jij_sum + num_Jij_per_cycle[j])
                        latency_per_spin["latency_read_sj"][i] = latency_per_spin["latency_read_sj"][i] + \
                                                                 len(set([math.floor(x / sram_setting["bw"]) for x in
                                                                          sj_ID_sorted[i][
                                                                          num_Jij_sum:cur_num_Jij_sum]]))
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

        latency_model = sum(latency_collect.values())
        energy_model = power_model * latency_model * t_clk
        logging.info(f"Benchmark: {benchmark}, Latency (model): {latency_model} cycles, Latency (reported): {latency} cycles, Energy (model): {energy_model} nJ, Energy (reported): {energy} nJ")
        benchmark_dict[benchmark]["energy_model"] = energy_model
        benchmark_dict[benchmark]["latency_model"] = latency_model
        benchmark_dict[benchmark]["latency_breakdown"] = latency_collect
    return benchmark_dict

if __name__ == "__main__":
    """
    validating the modeling results to PRIM CAEFA (HPCA'24)
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    plot_results_in_bar_chart_with_breakdown(benchmark_dict=validation_to_prim_caefa(), output_file="output/PRIM-CAEFA.png", text_type="absolute",
                                             with_latency_breakdown=True, latency_normalize=False)