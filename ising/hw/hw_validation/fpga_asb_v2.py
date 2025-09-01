import logging
import math
from api import plot_results_in_bar_chart


def validation_to_fpga_asb_v2():
    """
    validating the modeling results to FPGA-aSB Version 2 (IEEE Access'24), FPGA: Intel Arria10 (20nm)
    """
    """ validate to Table 2 in the paper, benchmark: all-to-all graphs, size: 2048-32768 spins, w pres: 1 bit """
    # HW settings
    nj_per_pe = 0.0257492  # 108000/(2048*2048), extracted from the paper
    # Benchmark settings
    benchmark_dict = {
        # key: K(num_spins)_C(num_cores). Note: in this work, K2000 has 2048 spins.
        # latency_comm: core-to-core latency; latency_comp: pipeline latency; freq_mhz: clock frequency (MHz)
        "K2000_C2": {
            "num_spins": 2048,
            "j_matrix_size": 2048 * 2048,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 2,
            "num_pe_rows": 1024,
            "num_pe_cols": 32,
            "latency_comm": 177,
            "latency_comp": 81,
            "freq_mhz": 281,
            "latency": 290,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K4000_C4": {
            "num_spins": 4096,
            "j_matrix_size": 4096 * 4096,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 4,
            "num_pe_rows": 1024,
            "num_pe_cols": 32,
            "latency_comm": 177,
            "latency_comp": 81,
            "freq_mhz": 281,
            "latency": 467,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K8000_C8": {
            "num_spins": 8192,
            "j_matrix_size": 8192 * 8192,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 8,
            "num_pe_rows": 1024,
            "num_pe_cols": 32,
            "latency_comm": 177,
            "latency_comp": 81,
            "freq_mhz": 281,
            "latency": 820,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K4000_C2": {
            "num_spins": 4096,
            "j_matrix_size": 4096 * 4096,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 2,
            "num_pe_rows": 2048,
            "num_pe_cols": 16,
            "latency_comm": 181,
            "latency_comp": 80,
            "freq_mhz": 301,
            "latency": 391,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K8000_C4": {
            "num_spins": 8192,
            "j_matrix_size": 8192 * 8192,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 4,
            "num_pe_rows": 2048,
            "num_pe_cols": 16,
            "latency_comm": 181,
            "latency_comp": 80,
            "freq_mhz": 301,
            "latency": 647,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K16000_C8": {
            "num_spins": 16384,
            "j_matrix_size": 16384 * 16384,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 8,
            "num_pe_rows": 2048,
            "num_pe_cols": 16,
            "latency_comm": 181,
            "latency_comp": 80,
            "freq_mhz": 301,
            "latency": 1160,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K8000_C2": {
            "num_spins": 8192,
            "j_matrix_size": 8192 * 8192,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 2,
            "num_pe_rows": 4096,
            "num_pe_cols": 8,
            "latency_comm": 177,
            "latency_comp": 87,
            "freq_mhz": 303,
            "latency": 1111,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K16000_C4": {
            "num_spins": 16384,
            "j_matrix_size": 16384 * 16384,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 4,
            "num_pe_rows": 4096,
            "num_pe_cols": 8,
            "latency_comm": 177,
            "latency_comp": 87,
            "freq_mhz": 303,
            "latency": 2135,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K32000_C8": {
            "num_spins": 32768,
            "j_matrix_size": 32768 * 32768,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 8,
            "num_pe_rows": 4096,
            "num_pe_cols": 8,
            "latency_comm": 177,
            "latency_comp": 87,
            "freq_mhz": 303,
            "latency": 4183,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "K16000_C2": {
            "num_spins": 16384,
            "j_matrix_size": 16384 * 16384,
            "num_iterations": 1,
            "w_pres": 1,
            "x_pres": 16,
            "p_pres": 16,
            "num_cores": 2,
            "num_pe_rows": 8192,
            "num_pe_cols": 4,
            "latency_comm": 167,
            "latency_comp": 101,
            "freq_mhz": 275,
            "latency": 4197,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
    }
    # calculating the performance metrics
    for benchmark, info in benchmark_dict.items():
        num_spins = info["num_spins"]
        j_matrix_size = info["j_matrix_size"]
        num_cores = info["num_cores"]
        num_pe_rows = info["num_pe_rows"]
        num_pe_cols = info["num_pe_cols"]
        latency_comm = benchmark_dict[benchmark]["latency_comm"]  # core-to-core latency
        latency_comp = benchmark_dict[benchmark][
            "latency_comp"
        ]  # latency to go through the pipeline within a core
        energy = benchmark_dict[benchmark]["energy"]  # reported energy (not valid)
        latency = benchmark_dict[benchmark]["latency"]  # reported latency
        # calculating the latency

        latency_compelem = num_spins / (num_cores * num_pe_cols)
        if latency_comm <= latency_compelem:  # compute-bounded region
            mode = "A"
        elif (
            latency_compelem < latency_comm <= 2 * latency_compelem
        ):  # mix region (communication-bounded in first calculation, compute-bounded afterwards)
            mode = "B"
        else:  # core-to-core communication bounded region
            mode = "C"
        if mode == "A":
            num_phases = num_spins / num_pe_rows
            latency_model = num_phases * latency_compelem + latency_comp
        elif mode == "B":
            num_phases = num_spins / num_pe_rows
            latency_model = (
                (num_phases - 1) * latency_compelem + latency_comm + latency_comp
            )
        else:
            n_hop = math.ceil((num_cores - 1) / 2)
            n_last_telem = 1 if num_cores % 2 == 0 else 2
            latency_model = (
                n_hop * latency_comm + n_last_telem * latency_compelem + latency_comp
            )
        # calculating the energy
        energy_model = nj_per_pe * j_matrix_size  # nJ
        logging.info(
        f"Benchmark: {benchmark}, Latency (model): {latency_model} cycles, "
        f"Latency (reported): {latency} cycles, "
        f"Energy (model): {energy_model} nJ, Energy (reported): {energy} nJ"
        )
        benchmark_dict[benchmark]["energy_model"] = energy_model
        benchmark_dict[benchmark]["latency_model"] = latency_model
    return benchmark_dict

    pass


if __name__ == "__main__":
    """
    validating the modeling results to FPGA-aSB (FPL'19)
    """
    logging_level = logging.INFO  # logging level
    logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=logging_level, format=logging_format)
    plot_results_in_bar_chart(
        validation_to_fpga_asb_v2(),
        output_file="output/fpga_asb_v2.png",
        text_type="absolute",
    )
