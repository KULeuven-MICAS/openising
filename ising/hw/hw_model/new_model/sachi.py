import logging
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import yaml

def sachi_hw_model(
        hw_model,
        workload,
        mapping,
):
    """
    The entire sachi hw architecture is shown below:
                bw_dram              sram_bw
    | DRAM |   ------->   | SRAM |   ------>   | Compute Memory |   ------>   | Compute Logics |

    For loop dimension definitions: i+1[t][j] += i[t][j] * w[j][l]
    - t: trail
    - it: iteration
    - i: spin
    - j: degree
    """
    """ The energy of updating adjacent matrixs IS considered in this model """
    """ There is no energy of annealing (random flipping, threshold comparison), as SACHI uses HNN """
    encoding_scheme = hw_model["operational_array"].get("encoding", "full-matrix")
    if encoding_scheme not in ["full-matrix", "triangular", "coordinate", "neighbor"]:
        raise ValueError(f"Unsupported encoding scheme: {encoding_scheme}")
    # scale the num_cores, as the sachi needs to store each spin twice
    num_cores_hw = hw_model["operational_array"]["sizes"][2]
    # if encoding_scheme == "neighbor":
    #     assert num_cores_hw > 1, "The number of cores must be larger than 1 for the neighbor encoding."
    #     num_cores_hw = num_cores_hw // 2
    #     hw_model["operational_array"]["sizes"][2] = num_cores_hw

    # extract the hardware dimension size
    hw_dim_sizes = {}
    for dim, size in zip(
            hw_model["operational_array"]["dimensions"],
            hw_model["operational_array"]["sizes"],
    ):
        hw_dim_sizes[dim] = size
    mac_count = np.prod([size for size in hw_dim_sizes.values()], dtype=np.int64)

    # extract the workload dimension size
    workload_dim_sizes = {}
    for dim, size in zip(
            workload["loop_dims"],
            workload["loop_sizes"],
    ):
        workload_dim_sizes[dim] = size
    workload_dim_sizes["T"] = workload["num_trails"]
    workload_dim_sizes["IT"] = workload["num_iterations"]

    if encoding_scheme not in ["full-matrix", "triangular"]:
        workload_dim_sizes["J"] = workload["average_degree"]

    if encoding_scheme == "full-matrix":  # dense, without compression
        bit_per_weight = workload["operand_precision"]["W"]
    elif encoding_scheme == "triangular":  # triangular, without compression
        bit_per_weight = workload["operand_precision"]["W"] // 2
    elif encoding_scheme == "coordinate":  # csr
        bit_per_weight = workload["operand_precision"]["W"] + math.ceil(math.log2(workload_dim_sizes["J"]))
    elif encoding_scheme == "neighbor":  # sachi's custom encoding
        bit_per_weight = workload["operand_precision"]["W"] + workload["operand_precision"]["I"]

    # calculate the total area
    area_collect = {}
    for component, spec in hw_model["memories"].items():
        served_dims = spec["served_dimensions"]
        repeat_count = 1
        for dim, size in hw_dim_sizes.items():
            if dim not in served_dims:
                repeat_count *= size
                if dim == "D3" and encoding_scheme == "neighbor":
                    repeat_count = repeat_count * 2  # this dim is halved beforehand in neighbor encoding
        area_collect[component] = spec["area"] * repeat_count
    area_collect["mac"] = hw_model["operational_array"]["mac_area"] * mac_count
    if workload.get("with_bias", False):
        area_collect["add"] = hw_model["operational_array"]["add_area"] * mac_count
    else:
        area_collect["add"] = 0
    area_collect["compare"] = hw_model["operational_array"]["compare_area"] * mac_count
    if encoding_scheme == "neighbor":
        area_collect["mac"] *= 2
        area_collect["add"] *= 2
        area_collect["compare"] *= 2
    total_area = sum([area for area in area_collect.values()])

    # extract spatial mapping hint
    if "spatial_mapping_hint" in mapping:
        spatial_mapping_hint: dict = mapping["spatial_mapping_hint"]
    else:
        spatial_mapping_hint: dict = {}

    # derive the mapping: spatial
    parfor_hw: dict = {key: 1 for key in hw_dim_sizes.keys()}
    parfor_sw: dict = {key: 1 for key in workload_dim_sizes.keys()}
    cim_memory_size: int = hw_model["memories"]["cim_memory"]["size"]
    cim_memory_bandwidth: int = hw_model["memories"]["cim_memory"]["bandwidth"]
    cim_memory_depth = cim_memory_size / cim_memory_bandwidth
    # constrain the parfor size by hardware/workload dimension sizes
    for d, item in spatial_mapping_hint.items():
        if (d == "D3"):
            parfor_hw[d] = min(hw_dim_sizes[d], workload_dim_sizes[item]/parfor_sw[item])
        elif (d == "D1"):
            parfor_hw[d] = 1  # fix D1 to 1 to match SACHI's design
        else:
            parfor_hw[d] = min(hw_dim_sizes[d], workload_dim_sizes[item]/parfor_sw[item])
        parfor_hw[d] = max(1, parfor_hw[d]) # ensure at least 1
    # constrain the parfor size by the lowest memory bandwidth
    for dim_hw in ["D2", "D1"]:
        parfor_size_hw = parfor_hw[dim_hw]
        if dim_hw == "D2":
            allowed_parfor_by_memory = max(1, cim_memory_bandwidth / bit_per_weight)
        else:
            allowed_parfor_by_memory = max(1, cim_memory_bandwidth / bit_per_weight / parfor_hw["D2"])
        if parfor_size_hw > allowed_parfor_by_memory:
            parfor_hw[dim_hw] = allowed_parfor_by_memory

    # finalize the parfor sizes
    for dim_hw in parfor_hw.keys():
        parfor_hw[dim_hw] = int(parfor_hw[dim_hw])

    for d in spatial_mapping_hint.keys():        
        for key in parfor_sw.keys():
            if key == spatial_mapping_hint[d]:
                parfor_sw[key] *= parfor_hw[d]

    # calculate the left workload loop size
    left_workload_dim_size: dict = copy.deepcopy(workload_dim_sizes)
    for d, item in spatial_mapping_hint.items():
        try:
            left_workload_dim_size[item] /= parfor_hw[d]
        except ZeroDivisionError:
            breakpoint()
    for d, value in left_workload_dim_size.items(): # round up
        left_workload_dim_size[d] = math.ceil(value)

    # derive the mapping: temporal
    mem_sizes_bit: dict = {}
    for component, spec in hw_model["memories"].items():
        served_operands = spec["operands"]
        if "I2" not in served_operands:
            continue
        served_dims = spec["served_dimensions"]
        repeat_count = 1
        for dim, size in hw_dim_sizes.items():
            if dim not in served_dims:
                repeat_count *= size
        mem_sizes_bit[component] = spec["size"] * repeat_count
    temfor_sw: list = [[key, value] for key, value in left_workload_dim_size.items()]
    temfor_hw: dict = {key: [] for key in mem_sizes_bit.keys()}

    unallocated_loops_sw = copy.deepcopy([v for v in temfor_sw if v[0] in ["I", "J"]])  # exclude T and IT
    allocated_loops_total = []
    for mem, size_bit in mem_sizes_bit.items():
        if unallocated_loops_sw == []:
            break
        allocated_loops = []
        if mem == "cim_memory":
            # special case for sachi cim memory
            allowed_loop_size = cim_memory_depth
        else:
            # calculate the minimal required mem size
            mem_sizes_weight_bit_min = (
                bit_per_weight
                * math.prod(parfor_hw.values())
                * math.prod([value for key, value in allocated_loops_total])
            )  # for weight
            mem_sizes_bias_bit_min = (
                workload["operand_precision"]["H"]
            * math.prod([value for key, value in parfor_sw.items() if key in ["I"]])
            * math.prod([value for key, value in allocated_loops_total])
            )  # for bias
            mem_sizes_bit_min = mem_sizes_weight_bit_min + mem_sizes_bias_bit_min
            allowed_loop_size = mem_sizes_bit[mem] / mem_sizes_bit_min
        if allowed_loop_size > 1:
            for idx in range(len(unallocated_loops_sw) - 1, -1, -1):
                if unallocated_loops_sw[idx][1] <= allowed_loop_size:
                    allocated_loops = [unallocated_loops_sw[idx]] + allocated_loops
                    allowed_loop_size /= unallocated_loops_sw[idx][1]
                    if idx == 0:  # all the loops are allocated
                        unallocated_loops_sw = []
                else:
                    allocated_loops = [
                        (unallocated_loops_sw[idx][0], int(allowed_loop_size))
                    ] + allocated_loops
                    unallocated_loops_sw[idx] = (
                        unallocated_loops_sw[idx][0],
                        unallocated_loops_sw[idx][1] / int(allowed_loop_size),
                    )
                    unallocated_loops_sw = unallocated_loops_sw[: idx + 1]
                    break
        temfor_hw[mem] = allocated_loops
        allocated_loops_total = allocated_loops + allocated_loops_total

    # calculate the top-level memory idx
    mem_list = [key for key in mem_sizes_bit.keys()]
    mem_bw_list = [hw_model["memories"][key]["bandwidth"] for key in mem_list]

    top_mem_idx = len(mem_list) - 1
    for mem_idx in range(len(mem_list)-1, -1, -1):
        if temfor_hw[mem_list[mem_idx]] == []:
            top_mem_idx = top_mem_idx - 1
        else:
            break

    # calculate the latency
    # there are two steps in each iteration: computation and spin updating
    # (i.e., packet updating/adjacent matrix updating)
    # latency of step one: comptutation
    ideal_latency = math.prod([value for key, value in temfor_sw])
    latency_collect: dict = {"compute": ideal_latency}
    latency_collect_breakdown: dict = {"compute": ideal_latency}
    access_collect: dict = {}
    parfor_size = math.prod([value for key, value in parfor_sw.items()])
    parfor_size_bias = math.prod([value for key, value in parfor_sw.items() if key in ["I"]])
    memory_double_buffering: bool = hw_model["operational_array"]["memory_double_buffering"]
    tmfor_size_total = math.prod(
        [value for key, value in temfor_sw]
    )  # consider i
    for mem_idx in range(top_mem_idx + 1):
        # tmfor loops below the current mem level
        tmfor_size_lower = math.prod(
            [value for mem in mem_list[:mem_idx] for key, value in temfor_hw[mem]]
        )
        tmfor_size_lower_bias = math.prod(
            [value for mem in mem_list[:mem_idx] for key, value in temfor_hw[mem] if key in ["I"]]
        )
        tmfor_size_upper = (
            tmfor_size_total / tmfor_size_lower
        )  # tmfor loops above and including the current mem level

        # calculate the write latency of tmfor_size_lower: wr_from_high
        if mem_idx == top_mem_idx:
            cycles_wr_per_tile = 0
        elif mem_list[mem_idx] == "cim_memory" and encoding_scheme == "neighbor":  # compute_memory
            cycles_wr_per_tile = 2  # 2: sachi stores the spins twice
        else:
            bit_per_tile = (tmfor_size_lower * parfor_size * bit_per_weight) + (tmfor_size_lower_bias * parfor_size_bias * workload["operand_precision"]["H"])
            cycles_wr_per_tile = bit_per_tile / (mem_bw_list[mem_idx])
            cycles_wr_per_tile = math.ceil(cycles_wr_per_tile)
        cycles_wr_from_high = math.ceil(cycles_wr_per_tile * tmfor_size_upper)

        # calculate the read latency of tmfor_size_lower: rd_to_low
        if mem_idx == 0:
            cycles_rd_per_tile = tmfor_size_lower
        else:
            bit_per_tile = (tmfor_size_lower * parfor_size * bit_per_weight) + (tmfor_size_lower_bias * parfor_size_bias * workload["operand_precision"]["H"])
            cycles_rd_per_tile = bit_per_tile / (mem_bw_list[mem_idx])
            cycles_rd_per_tile = math.ceil(cycles_rd_per_tile)
        cycles_rd_to_low = math.ceil(cycles_rd_per_tile * tmfor_size_upper)

        # calculate the write latency of tmfor_size_lower: wr_from_low
        cycles_wr_from_low = 0  # weights/biases never write back

        # calculate the read latency of tmfor_size_lower: rd_to_high
        cycles_rd_to_high = 0  # weights/biases never read back

        latency_collect_breakdown[mem_list[mem_idx]] = {
            "wr_from_high": cycles_wr_from_high,
            "rd_to_low": cycles_rd_to_low,
            "wr_from_low": cycles_wr_from_low,
            "rd_to_high": cycles_rd_to_high,
        }
        # calculate the latency, considering double buffer
        if memory_double_buffering:
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
    if encoding_scheme == "neighbor":
        # in this step, all the data must be read out from the top memory and write back to the top memory.
        data_size_bit = (
            workload_dim_sizes["I"] * workload_dim_sizes["J"] * bit_per_weight
        )
        for mem_idx in [top_mem_idx]:
            if (
                mem_idx == 0
            ):  # the data within the compute memory may not be densely stored (spatial utilization on D2 < 1)
                cycles_rd_spin_updating = latency_collect_breakdown[mem_list[mem_idx]]["rd_to_low"]
                cycles_wr_spin_updating = cycles_rd_spin_updating
            else:
                num_iterations = workload_dim_sizes["IT"]
                num_trails = workload_dim_sizes["T"]
                cycles_rd_spin_updating = math.ceil(data_size_bit / mem_bw_list[mem_idx]) * num_iterations * num_trails
                cycles_wr_spin_updating = math.ceil(data_size_bit / mem_bw_list[mem_idx]) * num_iterations * num_trails
            access_spin_updating: dict = {
                "wr": cycles_wr_spin_updating,
                "rd": cycles_rd_spin_updating,
            }
            if memory_double_buffering and mem_idx != 0:
                cycles_spin_updating = max(cycles_rd_spin_updating, cycles_wr_spin_updating)
            else:
                cycles_spin_updating = cycles_rd_spin_updating + cycles_wr_spin_updating
    else:
        cycles_spin_updating = 0
        access_spin_updating: dict = {"wr": ideal_latency, "rd": 0}
    # latency of step three: initial on-loading
    # note: this on-loading latency is zero if the top memory is dram
    bias_bit_total = workload["operand_precision"]["H"] * workload_dim_sizes["I"]
    if encoding_scheme == "full-matrix":
        weight_bit_total = workload["operand_precision"]["W"] * workload_dim_sizes["I"] * workload_dim_sizes["J"]
    elif encoding_scheme == "triangular":
        weight_bit_total = (workload["operand_precision"]["W"] // 2) * workload_dim_sizes["I"] * workload_dim_sizes["J"]
    elif encoding_scheme == "coordinate":
        weight_bit_total = (workload["operand_precision"]["W"] + math.log2(workload_dim_sizes["J"])) * workload_dim_sizes["I"] * workload_dim_sizes["J"]
    elif encoding_scheme == "neighbor":
        weight_bit_total = workload["operand_precision"]["W"] * workload_dim_sizes["I"] * workload_dim_sizes["J"]
    else:
        raise ValueError(f"Unsupported encoding scheme: {encoding_scheme}")
    if encoding_scheme == "neighbor":
        spin_bit_total = workload["operand_precision"]["I"] * workload_dim_sizes["I"] * workload_dim_sizes["J"]
    else:
        spin_bit_total = workload["operand_precision"]["I"] * workload_dim_sizes["I"]
    if workload["problem_specific_weight"] is False:
        weight_bit_total = 0 # weights are the same for different problems, so no on-loading
    onloading_bit_total = (bias_bit_total + weight_bit_total + spin_bit_total)
    onloading_latency_collect: dict = {}
    for mem_idx in range(top_mem_idx, len(mem_list)):
        if mem_idx != top_mem_idx:
            cycles_onloading_rd = math.ceil(onloading_bit_total / mem_bw_list[mem_idx])
        else:
            cycles_onloading_rd = 0
        if mem_idx == len(mem_list) - 1:
            cycles_onloading_wr = 0
        elif mem_list[mem_idx] == "cim_memory" and encoding_scheme == "neighbor":  # compute_memory
            cycles_onloading_wr = temfor_hw[mem_list[mem_idx]][0][1]
        else:
            cycles_onloading_wr = math.ceil(onloading_bit_total / mem_bw_list[mem_idx])
        onloading_latency_collect[mem_list[mem_idx]] = {
            "rd": cycles_onloading_rd,
            "wr": cycles_onloading_wr,
        }
    onloading_latency = max([value for spec in onloading_latency_collect.values() for key, value in spec.items()])
    # calculate the total latency
    latency_system = max(list(latency_collect.values())) + cycles_spin_updating + onloading_latency
    latency_system_breakdown = {
        "computation": latency_collect["compute"],
        "spin_updating": cycles_spin_updating,
        "on_loading": onloading_latency,
    }
    if max(list(latency_collect.values())) != latency_collect["compute"]:
        latency_system_breakdown["memory"] = max(list(latency_collect.values())) - latency_collect["compute"]

    # calculate the energy of step one: computation
    mac_energy = hw_model["operational_array"]["mac_energy"] * parfor_size * ideal_latency
    if workload.get("with_bias", False):
        add_energy = hw_model["operational_array"]["add_energy"] * parfor_size_bias * ideal_latency
    else:
        add_energy = 0
    compare_energy = hw_model["operational_array"]["compare_energy"] * parfor_size_bias * ideal_latency
    energy_compute_total = mac_energy + add_energy + compare_energy
    energy_collect: dict = {
        "mac": mac_energy,
        "add": add_energy,
        "compare": compare_energy,
    }
    mem_energy_total = 0
    for mem_idx in range(top_mem_idx + 1):
        mem_energy_per_wr = hw_model["memories"][mem_list[mem_idx]]["w_cost"]
        mem_energy_per_rd = hw_model["memories"][mem_list[mem_idx]]["r_cost"]
        mem_access = access_collect[mem_list[mem_idx]]
        mem_energy_wr = mem_energy_per_wr * mem_access["wr"]
        mem_energy_rd = mem_energy_per_rd * mem_access["rd"]
        mem_energy = mem_energy_wr + mem_energy_rd
        energy_collect[mem_list[mem_idx]] = {
            "wr": mem_energy_wr,
            "rd": mem_energy_rd,
        }
        mem_energy_total += mem_energy
    # calculate the energy of step two: spin updating
    if encoding_scheme == "neighbor":
        for mem_idx in [top_mem_idx]:
            mem_energy_per_wr = hw_model["memories"][mem_list[mem_idx]]["w_cost"]
            mem_energy_per_rd = hw_model["memories"][mem_list[mem_idx]]["r_cost"]
            spin_updating_energy_wr = mem_energy_per_wr * access_spin_updating["wr"]
            spin_updating_energy_rd = mem_energy_per_rd * access_spin_updating["rd"]
    else:
        # spin is saved in the register file
        for mem, spec in hw_model["memories"].items():
            if "O" in spec["operands"]:
                regfile_energy_per_wr = spec["w_cost"]
                regfile_energy_per_rd = spec["r_cost"]
                break
        spin_updating_energy_wr = regfile_energy_per_wr * access_spin_updating["wr"]
        spin_updating_energy_rd = regfile_energy_per_rd * access_spin_updating["rd"]
        energy_collect["spin_updating"] = {
            "wr": spin_updating_energy_wr,
            "rd": spin_updating_energy_rd,
        }
    spin_updating_energy_total = spin_updating_energy_wr + spin_updating_energy_rd
    # calculate the energy of step three: initial on-loading
    onloading_energy_collect: dict = {}
    for mem_idx in range(top_mem_idx, len(mem_list)):
        mem_energy_per_wr = hw_model["memories"][mem_list[mem_idx]]["w_cost"]
        mem_energy_per_rd = hw_model["memories"][mem_list[mem_idx]]["r_cost"]
        mem_energy_wr = mem_energy_per_wr * onloading_latency_collect[mem_list[mem_idx]]["wr"]
        mem_energy_rd = mem_energy_per_rd * onloading_latency_collect[mem_list[mem_idx]]["rd"]
        onloading_energy_collect[mem_list[mem_idx]] = {
            "wr": mem_energy_wr,
            "rd": mem_energy_rd,
        }
        energy_collect["onloading"] = onloading_energy_collect
    onloading_energy_total = sum([v["wr"] + v["rd"] for v in onloading_energy_collect.values()])
    energy_system = energy_compute_total + mem_energy_total + spin_updating_energy_total + onloading_energy_total
    energy_system_breakdown = {
        "computation": energy_compute_total,
        "memory": mem_energy_total,
        "spin_updating": spin_updating_energy_total,
        "on_loading": onloading_energy_total,
    }

    # calculate metrics
    time_to_solution = latency_system * hw_model["operational_array"]["tclk"]  # ns
    energy_to_solution = energy_system  # pJ
    num_op = workload["num_trails"] * workload["num_iterations"] * np.prod(workload["loop_sizes"]) * 2  # consider add and mac
    tops = num_op / time_to_solution / 1e3  # tera operations per second
    topsw = num_op / energy_to_solution  # operations per joule
    topsmm2 = tops / total_area  # tera operations per second per mm^2

    logging.info(f"Total Area (mm^2): {total_area:.2f}")
    logging.info(f"Time to Solution (ns): {time_to_solution:.2f}")
    logging.info(f"Energy to Solution (pJ): {energy_to_solution:.2f}")
    logging.info(f"TOPS: {tops:.2f}")
    logging.info(f"TOPSW: {topsw:.2f}")
    logging.info(f"TOPS/mm^2: {topsmm2:.2f}")

    latency_collect["spin_updating"] = cycles_spin_updating
    latency_collect["on_loading"] = onloading_latency
    req_sram_size = bit_per_weight * workload_dim_sizes["I"] * workload_dim_sizes["J"] + workload["operand_precision"]["H"] * workload_dim_sizes["I"]

    # interpret breakdown for plotting
    latency_system_breakdown_plot = {
        "mac": latency_system_breakdown["computation"],
        "spin_updating": latency_system_breakdown["spin_updating"],
    }
    if "memory" not in latency_system_breakdown:
        latency_system_breakdown_plot["sram"] = 0
        latency_system_breakdown_plot["dram"] = latency_system_breakdown["on_loading"]
    elif latency_system_breakdown["on_loading"] == 0:
        latency_system_breakdown_plot["sram"] = 0
        latency_system_breakdown_plot["dram"] = latency_system_breakdown["memory"]
    else:
        latency_system_breakdown_plot["sram"] = latency_system_breakdown["memory"]
        latency_system_breakdown_plot["dram"] = latency_system_breakdown["on_loading"]
    if "dram" in energy_collect.keys():
        energy_dram_wo_onloading = np.sum([value for value in energy_collect["dram"].values()])
    else:
        energy_dram_wo_onloading = 0
    energy_system_breakdown_plot = {
        "mac": energy_system_breakdown["computation"],
        "spin_updating": energy_system_breakdown["spin_updating"],
        "sram": energy_system_breakdown["memory"] - energy_dram_wo_onloading,
        "dram": energy_dram_wo_onloading + energy_system_breakdown["on_loading"],
    }

    cme = {
        "req_sram_size_bit": req_sram_size,
        "total_area_mm2": total_area,
        "time_to_solution": time_to_solution,
        "cycles_to_solution": latency_system,
        "energy_to_solution": energy_to_solution,
        "tops": tops,
        "topsw": topsw,
        "topsmm2": topsmm2,
        "latency_breakdown": latency_system_breakdown,
        "latency_breakdown_further": latency_collect,
        "energy_breakdown": energy_system_breakdown,
        "area_breakdown": area_collect,
        "latency_breakdown_plot": latency_system_breakdown_plot,
        "energy_breakdown_plot": energy_system_breakdown_plot,
        "parfor_hw": parfor_hw,
        "parfor_sw": parfor_sw,
        "temfor_hw": temfor_hw,
        "temfor_sw": temfor_sw,
        "workload_dim_sizes": workload_dim_sizes,
        "hw_dim_sizes": hw_dim_sizes,
        "bit_per_weight": bit_per_weight,
        "workload": workload,
        "hw_model": hw_model,
        "mapping": mapping,
    }

    return cme

if __name__ == "__main__":
    logging_format = ("%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO, format=logging_format)
    hw_model = yaml.safe_load(open("./inputs/hardware/sachi.yaml", 'r'))
    workload = yaml.safe_load(open("./inputs/workload/mc_500.yaml", 'r'))
    mapping = yaml.safe_load(open("./inputs/mapping/sachi.yaml", 'r'))
    sachi_hw_model(hw_model, workload, mapping)