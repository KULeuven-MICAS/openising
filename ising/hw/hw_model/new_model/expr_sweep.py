import logging
import yaml
import copy
from sachi import sachi_hw_model
import math
import csv
import pickle
import time
import tqdm
from get_cacti_cost import get_cacti_cost

if __name__ == "__main__":
    logging_format = ("%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    hw_model_org = yaml.safe_load(open("./inputs/hardware/sachi.yaml", 'r'))
    workload_org = yaml.safe_load(open("./inputs/workload/mc_500.yaml", 'r'))
    mapping_org = yaml.safe_load(open("./inputs/mapping/sachi.yaml", 'r'))
    component_list = ["mac", "cim", "sram", "dram"]
    component_tag_list = ["MAC", "CIM", "SRAM", "DRAM"]
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
        [200, 1, 16, True, False], # MIMO
    ]
    d1_in_list = [1]
    d2_in_list = [1, 5, 10, 50, 100, 500]
    macro_in_list = [2, 8, 16, 32, 64, 128]
    cim_depth_in_list = [10, 40, 80, 160, 320]
    sram_in_list = [1, 64, 160, 1024, 4096]  # in KB
    encoding_in_list = ["coordinate", "neighbor", "full-matrix"]
    benchmark_name_in_list = [f"{pb_spec[0]}" for pb_spec in pb_pool]
    
    # general settings
    # num_macros = 16

    results = [
        ["pb_size", "degree_density", "w_pres", "with_bias", "specific_w", "encoding", "sram_size_KB", "cim_depth", "num_macros", "D2", "D1",
         "cycles_to_solution", "energy_to_solution",
         "cycles_breakdown", "energy_breakdown",
         "tops", "topsw", "topsmm2", "full_result_dict"]
    ]
    with open("outputs/expr_sweep_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results[0])

    time_start = time.time()
    for pb_spec in pb_pool:
        pbar = tqdm.tqdm(total=len(encoding_in_list)*len(sram_in_list)*len(cim_depth_in_list)*len(macro_in_list)*len(d2_in_list)*len(d1_in_list), desc=f"PB size {pb_spec[0]}")
        for encoding_idx in range(len(encoding_in_list)):
            encoding = encoding_in_list[encoding_idx]
            for sram_size in sram_in_list:
                for cim_depth in cim_depth_in_list:
                    for macro_idx in range(len(macro_in_list)):
                        macro_in_list_current = macro_in_list[macro_idx:]
                        for d2 in d2_in_list:
                            for d1 in d1_in_list:
                                # unpack pb spec
                                pb_size, aver_density, weight_shared_precision, with_bias, problem_specific_weight = pb_spec
                                # copy basic settings
                                hw_model = copy.deepcopy(hw_model_org)
                                workload = copy.deepcopy(workload_org)
                                mapping = copy.deepcopy(mapping_org)
                                # set up parallelism sizes
                                num_macros = macro_in_list[macro_idx]
                                hw_model["operational_array"]["sizes"] = [d1, d2, num_macros]
                                # set up cim memory
                                if encoding == "coordinate":
                                    bit_per_weight = weight_shared_precision + math.log2(pb_size)
                                elif encoding == "neighbor":
                                    bit_per_weight = weight_shared_precision + 1
                                else:  # full-matrix
                                    bit_per_weight = weight_shared_precision
                                hw_model["memories"]["cim_memory"]["bandwidth"] = d2 * bit_per_weight
                                hw_model["memories"]["cim_memory"]["size"] = cim_depth * hw_model["memories"]["cim_memory"]["bandwidth"]  # in bits
                                _, hw_model["memories"]["cim_memory"]["area"], hw_model["memories"]["cim_memory"]["r_cost"], hw_model["memories"]["cim_memory"]["w_cost"] = get_cacti_cost(cacti_path='./cacti/cacti_master', tech_node=0.028,
                                                                                              mem_type='sram', mem_size_in_byte=hw_model["memories"]["cim_memory"]["size"]/8,
                                                                                              bw=hw_model["memories"]["cim_memory"]["bandwidth"])
                                # set up sram size
                                sram_size_in_KB = sram_size
                                hw_model["memories"]["sram_160KB"]["size"] = sram_size_in_KB * 1024 * 8  # in bits
                                _, hw_model["memories"]["sram_160KB"]["area"], hw_model["memories"]["sram_160KB"]["r_cost"], hw_model["memories"]["sram_160KB"]["w_cost"] = get_cacti_cost(cacti_path='./cacti/cacti_master', tech_node=0.028,
                                                                                              mem_type='sram', mem_size_in_byte=hw_model["memories"]["sram_160KB"]["size"]/8,
                                                                                              bw=hw_model["memories"]["sram_160KB"]["bandwidth"])
                                # set up encoding
                                hw_model["operational_array"]["encoding"] = encoding
                                # setup workload
                                workload["loop_sizes"] = [pb_size, pb_size]
                                workload["operand_precision"]["W"] = weight_shared_precision
                                workload["operand_precision"]["H"] = weight_shared_precision
                                workload["average_degree"] = aver_density * pb_size
                                workload["with_bias"] = with_bias
                                workload["problem_specific_weight"] = problem_specific_weight
                                # simulation
                                cme = sachi_hw_model(hw_model, workload, mapping)
                                # collect results
                                cycles_to_solution = cme["cycles_to_solution"]
                                energy_to_solution = cme["energy_to_solution"]
                                cycles_breakdown = cme["latency_breakdown_plot"]
                                energy_breakdown = cme["energy_breakdown_plot"]
                                tops = cme["tops"]
                                topsw = cme["topsw"]
                                topsmm2 = cme["topsmm2"]
                                # save results
                                new_output = [
                                    pb_size, aver_density, weight_shared_precision, with_bias, problem_specific_weight,
                                    encoding, sram_size_in_KB,  cim_depth, num_macros, d2, d1,
                                    cycles_to_solution, energy_to_solution,
                                    cycles_breakdown, energy_breakdown,
                                    tops, topsw, topsmm2, cme
                                ]
                                results.append(new_output)
                                # print(pb_size, aver_density, weight_shared_precision, with_bias, problem_specific_weight,
                                #       encoding, sram_size_in_KB,  cim_depth, num_macros, d2, d1,
                                #     cycles_to_solution, energy_to_solution,
                                #     tops, topsw, topsmm2)
                                # save results
                                with open("outputs/expr_sweep_results.csv", "a", newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerows([new_output])
                                pbar.update(1)
    # save results to pickle
    with open("outputs/expr_sweep_results.pkl", "wb") as f:
        pickle.dump(results, f)
    time_end = time.time()
    print(f"Total experiment time: {time_end - time_start} seconds")
    breakpoint()
