import logging
from api import plot_results_in_bar_chart

def validation_to_fpga_asb():
    """
    validating the modeling results to FPGA-aSB (FPL'19), FPGA: Intel Arria10 GX1150 (20nm)
    """
    """ validate to Table 1 in the paper, benchmark: all-to-all graphs, size: 2048/4096 spins, w pres: 1 bit """
    # HW settings
    num_cores = 8
    num_pe_rows = 32
    num_pe_cols = 32
    global_mem_access_latency = 48  # cycles, latency of writing x back to X' mem in Fig. 1 (lamda in the paper)
    nj_per_pe = 0.0257492  # 108000/(2048*2048), extracted from the paper
    # tclk = 1000 / 279  # ns (not used)
    # Benchmark settings
    benchmark_dict = {
        # latency [cycle]: reported latency per iteration, energy [nJ]: reported energy per iteration,
        # latency_model [cycle]: latency to be modeled, energy_model [nJ]: energy to be modeled
        "K2000": {"num_spins": 2048, "j_matrix_size": 2048 * 2048, "num_iterations": 25,
                  "w_pres": 1, "x_pres": 16, "p_pres": 16, "latency": 624, "energy": 108000,
                  "latency_model": 0, "energy_model": 0},
        "K4000": {"num_spins": 4096, "j_matrix_size": 4096*4096, "num_iterations": 25,
                  "w_pres": 1, "x_pres": 16, "p_pres": 16, "latency": 2224, "energy": 108000 * 4,
                  "latency_model": 0, "energy_model": 0},
    }

    # calculating the performance metrics
    for benchmark, info in benchmark_dict.items():
        num_spins = info["num_spins"]
        j_matrix_size = info["j_matrix_size"]
        energy = info["energy"]
        latency = info["latency"]
        # calculating the latency
        num_phases = num_spins / ( num_pe_rows * num_cores )
        latency_model = num_spins / num_pe_cols + num_phases * max(num_spins / num_pe_cols, num_pe_rows) \
            + global_mem_access_latency
        # calculating the energy
        energy_model = nj_per_pe * j_matrix_size  # nJ
        logging.info(f"Benchmark: {benchmark}, Latency (model): {latency_model} cycles, "
            f"Latency (reported): {latency} cycles, Energy (model): {energy_model} nJ, Energy (reported): {energy} nJ")
        benchmark_dict[benchmark]["energy_model"] = energy_model
        benchmark_dict[benchmark]["latency_model"] = latency_model
    return benchmark_dict

if __name__ == "__main__":
    """
    validating the modeling results to FPGA-aSB (FPL'19)
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    plot_results_in_bar_chart(validation_to_fpga_asb(), output_file="output/fpga_asb.png", text_type="absolute")
