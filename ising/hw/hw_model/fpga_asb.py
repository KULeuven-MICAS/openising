import logging
import matplotlib.pyplot as plt
import copy
import math


def fpga_asbv2_hw_model(
    # bw: bit width @ DDR5, size_gb: mem size in GB
    dram_setting: dict = {
        "size_gb": 800,
        "bw": 128,
        "pj_per_bit": 3.7,
        "double_buf": True,
    },
    # size_kb: mem size in KB, bw: bit width; area: mm2; @TSMC28nm
    sram_setting: dict = {
        "size_kb": 160,
        "bw": 1024,
        "pj_w_per_bit": 0.04987,
        "pj_r_per_bit": 0.0587,
        "area": 0.04975,
        "double_buf": True,
    },
    # bw: bit width; area: mm2; @TSMC28nm
    local_j_mem_setting: dict = {
        "size_kb": 80 // 8,
        "bw": 32768,
        "pj_w_per_bit": 0.0157,
        "pj_r_per_bit": 0.0089,
        "area": 0.03636,
        "double_buf": True,
    },
    num_spins_per_core: int = 1024,  # num of spins each core has, affect the local x/p mem size
    pe_size: dict = {
        "d1": 32,
        "d2": 1024,
        "d3": 2,
    },  # d1: num_pe_cols, d2: num_pe_rows, d3: num_cores
    pj_compute_per_bit: float = 1.25
    * ((28 / 45) ** 2)
    * ((0.9) ** 2),  # pJ/bit@@TSMC28nm, Vdd=0.9V, energy per gate
    mm2_compute_per_bit: float = 0.614 * 1e-6,  # mm2/bit, also mm2/gate
    tclk: int = 1,  # ns, manually defined
    benchmark_dict: dict = {
        "name": "Molecular Dynamics",
        "num_spins": 500,
        "num_degrees": 8,
        "num_iterations": 1,
        "w_pres": 2,
        "x_pres": 16,
        "p_pres": 16,
        "num_trails": 1,
    },
    mapping: dict = {"d1": "l", "d2": "j", "d3": "j"},
    fpga_mode: str = "v2",
    core2core_cycle: int = 1,  # cycles when fpga_mode is "v2", core-to-core communication latency
    # num_substeps: M factor in the algorithm
    updating_setting: dict = {"num_substeps": 2, "updating_parallelism_per_core": 1024},
):
    """
    The entire sachi hw architecture is shown below:
                bw_dram              sram_bw
    | DRAM |   ------->   | SRAM |   ------>   | local j/x/p mem |   ------>   | Compute Logics |

    For loop dimension definitions: i+1[t][j] += i[t][j] * w[j][l]
    - t: trail
    - i: iteration
    - j: spin
    - l: degree
    """
    # derive local x/p memory size (reg files)
    mm2_per_reg = 6 * 0.614 * 1e-6  # mm2 per reg file
    local_x_mem_size_bit = num_spins_per_core * benchmark_dict["x_pres"]  # per core
    local_p_mem_size_bit = num_spins_per_core * benchmark_dict["p_pres"]  # per core
    local_x_mem_area = (
        pe_size["d3"] * local_x_mem_size_bit * mm2_per_reg
    )  # total area (across cores)
    local_p_mem_area = (
        pe_size["d3"] * local_p_mem_size_bit * mm2_per_reg
    )  # total area (across cores)
    assert local_j_mem_setting["size_kb"] * 1024 * 8 >= local_x_mem_size_bit, (
        f"local j mem size ({local_j_mem_setting["size_kb"]} KB) is too small "
        f"to support the targeted spins per core {num_spins_per_core}"
    )
    # calculate the total area
    compute_area = pe_size["d1"] * pe_size["d2"] * pe_size["d3"] * mm2_compute_per_bit
    area_collect = {
        "sram": sram_setting["area"],
        "local_j_mem": local_j_mem_setting["area"],
        "local_x_mem": local_x_mem_area,
        "local_p_mem": local_p_mem_area,
        "compute": compute_area,
    }
    area_total = sum([area_collect[key] for key in area_collect.keys()])
    # derive the workload representation
    workload: dict = {
        "t": benchmark_dict["num_trails"],
        "i": benchmark_dict["num_iterations"],
        "j": benchmark_dict["num_spins"],
        "l": benchmark_dict["num_degrees"],
    }
    # derive the mapping: spatial
    parfor_sw: list = [(key, 1) for key in workload.keys()]
    left_workload = copy.deepcopy(workload)
    parfor_hw: dict = {
        "d1": min(workload[mapping["d1"]], pe_size["d1"]),  # num_pe_cols
        "d2": min(workload[mapping["d2"]], pe_size["d2"]),  # num_pe_rows
        "d3": 0,
    }  # num_cores
    for hw_dim in ["d1", "d2"]:
        sw_dim = mapping[hw_dim]
        left_workload[sw_dim] = math.ceil(left_workload[sw_dim] / parfor_hw[hw_dim])
    parfor_hw["d3"] = min(left_workload[mapping["d3"]], pe_size["d3"])  # num_cores
    left_workload[mapping["d3"]] = math.ceil(
        left_workload[mapping["d3"]] / parfor_hw["d3"]
    )
    for hw_dim in pe_size.keys():  # derive the parfor_sw
        sw_dim = mapping[hw_dim]
        for i, (key, value) in enumerate(parfor_sw):
            if key == sw_dim:
                parfor_sw[i] = (key, value * parfor_hw[hw_dim])
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
        "local_j_mem": local_j_mem_setting["size_kb"] * 1024 * 8,
    }
    mem_list = ["local_j_mem", "sram", "dram"]
    temfor_hw: dict = {key: [] for key in mem_list}  # create based on mem_list
    unallocated_loops = copy.deepcopy(
        temfor_sw[2:]
    )  # neglect the trail and iteration loops
    allocated_loops_total = []
    # derive the allowed loop size on j for local_j_mem
    # num_spins_per_core, parfor_sw[2]: j
    allowed_loop_size_on_j = min(
        num_spins_per_core / parfor_sw[2][1], unallocated_loops[0][1]
    )
    allocated_loops_on_j: tuple = ("j", allowed_loop_size_on_j)
    unallocated_loops[0] = (
        "j",
        math.ceil(unallocated_loops[0][1] / allowed_loop_size_on_j),
    )
    if unallocated_loops[0][1] == 1:
        unallocated_loops.pop(0)  # remove the allocated loop
    for mem in mem_list:
        if unallocated_loops == []:
            break
        allocated_loops = []
        if mem == "local_j_mem":
            allocated_loops.append(allocated_loops_on_j)
            # derive the allowed loop size on l
            mem_sizes_bit_min = (
                benchmark_dict["w_pres"]
                * math.prod(parfor_hw.values())
                * math.prod([value for key, value in allocated_loops_total])
            )  # for weight
            allowed_loop_size_on_l = (
                mem_sizes_bit[mem] / allowed_loop_size_on_j / mem_sizes_bit_min
            )
            if allowed_loop_size_on_l > 1:
                unallocated_loop_size_on_l = unallocated_loops[-1][1]  # l
                allocated_loops.append(
                    ("l", min(allowed_loop_size_on_l, unallocated_loop_size_on_l))
                )
                if allowed_loop_size_on_l >= unallocated_loop_size_on_l:
                    unallocated_loops.pop(-1)
                else:
                    unallocated_loops[-1] = (
                        "l",
                        math.ceil(unallocated_loops[-1][1] / allocated_loops[-1][1]),
                    )
            logging.info(
                f"Suitable j mem size (putting entire N*pr J in the mem) is: "
                f"{unallocated_loop_size_on_l * allowed_loop_size_on_j * mem_sizes_bit_min / 8 / 1024} KB, "
                f"curr: {local_j_mem_setting["size_kb"]} KB"
            )
        else:
            if (
                unallocated_loops[0][0] == "j" and unallocated_loops[0][1] > 1
            ):  # x is to be saved in current mem
                mem_sizes_bit_min = benchmark_dict["w_pres"] * math.prod(
                    parfor_hw.values()
                ) * math.prod(
                    [value for key, value in allocated_loops_total]
                ) + benchmark_dict[
                    "x_pres"
                ] * parfor_sw[
                    2
                ][
                    1
                ] * math.prod(
                    [value for key, value in allocated_loops_total if key == "j"]
                )  # for weight and x
                allowed_loop_size_on_j = mem_sizes_bit[mem] / mem_sizes_bit_min
                if allowed_loop_size_on_j > 1:
                    unallocated_loop_size_on_j = unallocated_loops[0][1]  # j
                    if allowed_loop_size_on_j >= unallocated_loop_size_on_j:
                        allocated_loops.append(("j", unallocated_loop_size_on_j))
                        unallocated_loops.pop(0)
                        # check if l can be allocated
                        # fixed mem size for x
                        mem_sizes_bit_for_x = (
                            unallocated_loop_size_on_j
                            * benchmark_dict["x_pres"]
                            * parfor_sw[2][1]
                            * math.prod(
                                [
                                    value
                                    for key, value in allocated_loops_total
                                    if key == "j"
                                ]
                            )
                        )
                        mem_sizes_bit_for_l = (
                            unallocated_loop_size_on_j
                            * benchmark_dict["w_pres"]
                            * math.prod(parfor_hw.values())
                            * math.prod([value for key, value in allocated_loops_total])
                        )  # for weight
                        allowed_loop_size_on_l = (
                            mem_sizes_bit[mem] - mem_sizes_bit_for_x
                        ) / mem_sizes_bit_for_l
                        if allowed_loop_size_on_l > 1:
                            unallocated_loop_size_on_l = unallocated_loops[-1][1]
                            allocated_loops.append(
                                (
                                    "l",
                                    min(
                                        allowed_loop_size_on_l,
                                        unallocated_loop_size_on_l,
                                    ),
                                )
                            )
                            if allowed_loop_size_on_l >= unallocated_loop_size_on_l:
                                unallocated_loops.pop(-1)
                            else:
                                unallocated_loops[-1] = (
                                    "l",
                                    math.ceil(
                                        unallocated_loops[-1][1]
                                        / allowed_loop_size_on_l
                                    ),
                                )
                    else:
                        allocated_loops.append(("j", math.ceil(allowed_loop_size_on_j)))
                        unallocated_loops[0] = (
                            "j",
                            math.ceil(unallocated_loops[0][1] / allowed_loop_size_on_j),
                        )
            else:  # x is not to be saved in current mem
                mem_sizes_bit_for_l = (
                    benchmark_dict["w_pres"]
                    * math.prod(parfor_hw.values())
                    * math.prod([value for key, value in allocated_loops_total])
                )  # for weight
                allowed_loop_size_on_l = mem_sizes_bit[mem] / mem_sizes_bit_for_l
                if allowed_loop_size_on_l > 1:
                    unallocated_loop_size_on_l = unallocated_loops[0][1]
                    allocated_loops.append(
                        ("l", min(allowed_loop_size_on_l, unallocated_loop_size_on_l))
                    )
                    if allowed_loop_size_on_l >= unallocated_loop_size_on_l:
                        unallocated_loops.pop(0)
                    else:
                        unallocated_loops[0] = (
                            "l",
                            math.ceil(unallocated_loops[0][1] / allowed_loop_size_on_l),
                        )
        temfor_hw[mem] = allocated_loops
        allocated_loops_total += allocated_loops
    # calculate the top-level memory idx
    top_mem_idx = len(mem_list) - 1
    for mem_idx in range(len(mem_list)):
        if temfor_hw[mem_list[mem_idx]] == []:
            top_mem_idx = mem_idx - 1
            break
    # calculate the top-level memory idx for x
    top_mem_idx_x = len(mem_list) - 1
    for mem_idx in range(len(mem_list)):
        if (
            temfor_hw[mem_list[mem_idx]] == []
            or temfor_hw[mem_list[mem_idx]][0][0] != "j"
        ):
            top_mem_idx_x = mem_idx - 1
            break
    # calculate the latency
    # there are two steps in each iteration: computation and spin updating
    # (i.e., packet updating/adjacent matrix updating)
    # latency of step one: comptutation
    mem_bw_list = [
        local_j_mem_setting["bw"] * pe_size["d3"],
        sram_setting["bw"],
        dram_setting["bw"],
    ]
    mem_dbuf_list = [
        local_j_mem_setting["double_buf"],
        sram_setting["double_buf"],
        dram_setting["double_buf"],
    ]
    ideal_latency = math.prod([value for key, value in temfor_sw])
    latency_collect: dict = {"compute": ideal_latency}
    access_collect: dict = {}
    access_collect_detail: dict = {}
    parfor_size = math.prod([value for key, value in parfor_sw])
    parfor_size_x = math.prod([value for key, value in parfor_sw if key == "j"])
    for mem_idx in range(top_mem_idx + 1):
        # tmfor loops below the current mem level
        tmfor_size_lower = math.prod(
            [value for mem in mem_list[:mem_idx] for key, value in temfor_hw[mem]]
        )
        # tmfor loops on x below the current mem level
        tmfor_size_x_lower = math.prod(
            [
                value
                for mem in mem_list[:mem_idx]
                for key, value in temfor_hw[mem]
                if key == "j"
            ]
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
        else:
            if (
                mem_idx != 0 and mem_idx < top_mem_idx_x
            ):  # transmit x. mem_idx != 0: not j mem
                cycles_wr_per_tile = (
                    (tmfor_size_lower * parfor_size * benchmark_dict["w_pres"])
                    + tmfor_size_x_lower * parfor_size_x * benchmark_dict["x_pres"]
                ) / (mem_bw_list[mem_idx])
            else:
                cycles_wr_per_tile = (
                    tmfor_size_lower * parfor_size * benchmark_dict["w_pres"]
                ) / (mem_bw_list[mem_idx])
            cycles_wr_per_tile = math.ceil(cycles_wr_per_tile)
        cycles_wr_from_high = math.ceil(cycles_wr_per_tile * tmfor_size_upper)
        # calculate the read latency of tmfor_size_lower: rd_to_low
        if (
            mem_idx != 0 and mem_idx < top_mem_idx_x
        ):  # transmit x. mem_idx != 0: not j mem
            cycles_rd_per_tile = (
                (tmfor_size_lower * parfor_size * benchmark_dict["w_pres"])
                + tmfor_size_x_lower * parfor_size_x * benchmark_dict["x_pres"]
            ) / (mem_bw_list[mem_idx])
        else:
            cycles_rd_per_tile = (
                tmfor_size_lower * parfor_size * benchmark_dict["w_pres"]
            ) / (mem_bw_list[mem_idx])
        cycles_rd_per_tile = math.ceil(cycles_rd_per_tile)
        cycles_rd_to_low = math.ceil(cycles_rd_per_tile * tmfor_size_upper)
        # calculate the write latency of tmfor_size_lower: wr_from_low
        if mem_idx == 0 or mem_idx > top_mem_idx:
            cycles_wr_per_tile = 0  # no write back for x
        else:  # transmit x
            cycles_wr_per_tile = (
                tmfor_size_x_lower * parfor_size_x * benchmark_dict["x_pres"]
            ) / (mem_bw_list[mem_idx])
            cycles_wr_per_tile = math.ceil(cycles_wr_per_tile)
        cycles_wr_from_low = math.ceil(cycles_wr_per_tile * tmfor_size_upper)
        # calculate the read latency of tmfor_size_lower: rd_to_high
        if mem_idx == top_mem_idx:
            cycles_rd_per_tile = 0
        elif mem_idx == 0 or mem_idx >= top_mem_idx:
            cycles_rd_per_tile = 0
        else:
            cycles_rd_per_tile = (
                tmfor_size_x_lower * parfor_size_x * benchmark_dict["x_pres"]
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
        access_collect_detail[mem_list[mem_idx]] = {
            "wr_from_high": cycles_wr_from_high,
            "wr_from_low": cycles_wr_from_low,
            "rd_to_low": cycles_rd_to_low,
            "rd_to_high": cycles_rd_to_high,
        }
    # latency of step one: communication (comm) for x_j (x' in the paper)
    if fpga_mode == "v1":
        separate_x_regfile_size = benchmark_dict["x_pres"] * workload["j"]  # x_pres * j
        cycle_per_tile = 1  # assume infinite bandwidth for regfiles
        cycles_comm = cycle_per_tile * math.prod([value for key, value in temfor_sw])
        latency_collect["comm"] = cycles_comm
        area_collect["comm"] = separate_x_regfile_size * mm2_per_reg
        area_total += area_collect["comm"]
    else:
        num_hops = math.ceil((pe_size["d3"] - 1) / 2)
        # x_pres * num_pe_cols * num_cores
        separate_x_regfile_size = (
            benchmark_dict["x_pres"] * pe_size["d1"] * pe_size["d3"]
        )
        # core2core_cycle * num_hops per iteration
        cycles_comm = core2core_cycle * num_hops
        # consider t and i
        cycles_comm *= math.prod(
            [value for key, value in temfor_sw if key in ["t", "i"]]
        )
        latency_collect["comm"] = cycles_comm
        area_collect["comm"] = separate_x_regfile_size * mm2_per_reg
        area_total += area_collect["comm"]

    # latency of step two: spin updating
    num_substeps = updating_setting["num_substeps"]  # M factor in the algorithm
    # assume infinite bandwidth for regfiles
    updating_parallelism_per_core = updating_setting["updating_parallelism_per_core"]
    if fpga_mode == "v1":
        assert temfor_hw["local_j_mem"][1][0] == "l"
        num_spins_to_update_per_core = parfor_hw[
            "d2"
        ]  # pipelined, see Fig. 3 in the paper
        cycles_per_spin_updating = math.ceil(
            num_spins_to_update_per_core * num_substeps / updating_parallelism_per_core
        )
        times_of_updating = math.ceil(
            math.prod([value for key, value in temfor_sw])
            / temfor_hw["local_j_mem"][1][1]
        )
        # check if the updating cycle can be hidden by the computation
        if (
            cycles_per_spin_updating <= temfor_hw["local_j_mem"][1][1]
        ):  # hidden by the computation
            cycles_spin_updating = 0
        else:
            cycles_spin_updating = (
                cycles_per_spin_updating - temfor_hw["local_j_mem"][1][1]
            ) * times_of_updating
    else:
        assert temfor_hw["local_j_mem"][0][0] == "j"
        num_spins_to_update_per_core = (
            parfor_hw["d2"] * temfor_hw["local_j_mem"][0][1]
        )  # num_pe_rows * j
        times_of_updating = math.ceil(
            math.prod([value for key, value in temfor_sw])
            / temfor_hw["local_j_mem"][0][1]
            / temfor_hw["local_j_mem"][1][1]
        )
        cycles_spin_updating = (
            math.ceil(
                num_spins_to_update_per_core
                * num_substeps
                / updating_parallelism_per_core
            )
            * times_of_updating
        )

    # calculate the energy of step one: computation
    compute_energy = (
        pj_compute_per_bit
        * parfor_size
        * benchmark_dict["w_pres"]
        * benchmark_dict["x_pres"]
        * ideal_latency
    )
    energy_collect: dict = {"compute": compute_energy}
    mem_energy_wr_per_bit = {
        "dram": dram_setting["pj_per_bit"],
        "sram": sram_setting["pj_w_per_bit"],
        "local_j_mem": local_j_mem_setting["pj_w_per_bit"],
    }
    mem_energy_rd_per_bit = {
        "dram": dram_setting["pj_per_bit"],
        "sram": sram_setting["pj_r_per_bit"],
        "local_j_mem": local_j_mem_setting["pj_r_per_bit"],
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
            else:
                energy_collect[mem][key] = (
                    access_collect[mem][key]
                    * mem_bw_list[mem_idx]
                    * mem_energy_rd_per_bit[mem]
                )
    # calculate the energy of step two: spin updating
    num_cores = pe_size["d3"]
    energy_spin_updating = (
        num_spins_to_update_per_core
        * num_cores
        * num_substeps
        * times_of_updating
        * pj_compute_per_bit
        * (benchmark_dict["x_pres"] + benchmark_dict["p_pres"])
    )
    energy_collect["energy_SU"] = energy_spin_updating
    # calculate the overall system latency and energy
    if fpga_mode == "v1":
        latency_system = max(
            list(latency_collect.values()), cycles_spin_updating
        )  # spin updating: pipelined
    else:
        latency_system = (
            max(list(latency_collect.values())) + cycles_spin_updating
        )  # spin updating: sequential

    energy_system = (
        sum([sum(energy_collect[mem].values()) for mem in mem_list[: top_mem_idx + 1]])
        + compute_energy
        + energy_spin_updating
    )
    # extra: for the ease of plotting results later, add cycles_spin_updating within the latency_collect
    latency_collect["spin_updating"] = cycles_spin_updating
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
    logging.info(
        f"Top mem: {mem_list[top_mem_idx]}, Top mem for x: {mem_list[top_mem_idx_x]}"
    )
    logging.info(
        f"System Spec: SRAM size: {sram_setting['size_kb']} KB, "
        f"Local J mem size: {local_j_mem_setting['size_kb']} KB, "
        f"Local X reg size: {local_x_mem_size_bit/8/1024} KB, Local P reg size: {local_p_mem_size_bit/8/1024} KB"
    )
    logging.info(
        f"Local X' reg size: {separate_x_regfile_size/8/1024} KB, PE size: {math.prod(pe_size.values())}, "
        f"updating parallelism per core: {updating_parallelism_per_core}, num spins per core: {num_spins_per_core}"
    )
    # return the performance: latency breakdown, energy breakdown, area breakdown, latency in cycles,
    # latency in ns, energy in pj, area in mm2
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
    fpga_mode: str = "v2",
    output_file: str = "output/fpga_asb.png",
):
    """plot the latency (left) in bar, and energy breakdown (right) in pie chart"""
    assert fpga_mode in ["v1", "v2"], f"Invalid fpga_mode: {fpga_mode}"
    # calculate the total latency, energy, area
    total_latency = (
        sum(latency_collect.values())
        if fpga_mode == "v1"
        else sum(
            [value for key, value in latency_collect.items() if key != "spin_updating"]
        )
        + latency_collect["spin_updating"]
    )
    total_energy = (
        energy_collect["compute"]
        + sum(
            [
                sum(energy_collect[mem].values())
                for mem in energy_collect.keys()
                if mem not in ["compute", "energy_SU"]
            ]
        )
        + energy_collect["energy_SU"]
    )
    total_area = sum(area_collect.values())
    # plotting the results
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    if fpga_mode == "v1":
        labels = [key for key, value in latency_collect.items()]
        sizes = [value for key, value in latency_collect.items()]
    else:
        labels = [
            key for key, value in latency_collect.items() if key != "spin_updating"
        ]
        sizes = [
            value for key, value in latency_collect.items() if key != "spin_updating"
        ]
        sizes_su = []
        for mem_idx in range(len(labels)):
            if mem_idx == sizes.index(max(sizes)):
                latency_su = [
                    value
                    for key, value in latency_collect.items()
                    if key == "spin_updating"
                ][0]
                sizes_su.append(latency_su)
            else:
                sizes_su.append(0)
    """ plotting the latency breakdown """
    ax[0].bar(labels, sizes, bottom=0, edgecolor="black", label="computation")
    if fpga_mode == "v2":
        ax[0].bar(
            labels,
            sizes_su,
            bottom=sizes,
            edgecolor="black",
            label="spin updating (SU)",
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
    # sum the energy of read and write, as the energy of read and write are the same
    sizes = (
        [energy_collect["compute"]]
        + [sum(energy_collect[mem].values()) for mem in labels[1:-1]]
        + [energy_collect["energy_SU"]]
    )
    ax[1].pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
        textprops={"weight": "bold"},
    )
    ax[1].axis("equal")
    ax[1].set_title(f"Energy [{round(total_energy, 2)} nJ]", weight="bold")
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
    ax[2].set_title(f"Area [{round(total_area, 2)} mm2]", weight="bold")
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
    fpga_mode = "v2"
    benchmark_dict = {
        "name": "Molecular Dynamics",
        "num_spins": 500,
        "num_degrees": 8,
        "num_iterations": 1,
        "w_pres": 2,
        "x_pres": 16,
        "p_pres": 16,
        "num_trails": 1,
    }
    benchmark_name = benchmark_dict["name"]
    """ experiment settings end """
    (
        latency_collect,
        energy_collect,
        area_collect,
        latency_system,
        latency_system_in_ns,
        energy_system,
        area_total,
    ) = fpga_asbv2_hw_model(benchmark_dict=benchmark_dict, fpga_mode=fpga_mode)
    plot_results_in_pie_chart(
        benchmark_name,
        latency_collect,
        energy_collect,
        area_collect,
        fpga_mode,
        output_file="output/fpga_asb.png",
    )
