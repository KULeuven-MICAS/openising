import logging
from api import plot_results_in_bar_chart


def validation_to_sachi():
    """
    validating the modeling results to SACHI (HPCA'24)
    validate to Fig 15 (b)(c) in SACHI, benchmark: TSP, Molecular Dynamics (King graph),
    size: 1k spins, w pres: 4bit
    it should be noted:
    the reported latency and energy do not include the memory access latency,
    even though the problem size exceeds the on-chip memory capacity
    """
    # HW settings
    num_cores = 16
    compute_memory_depth = 80
    # tclk = 5  # ns (not used)
    energy_per_spin_per_degree_per_bit = (
        5 / 4
    )  # pJ/bit@FreePDK45nm, Vdd=1V (extracted from the paper)
    # Benchmark settings
    benchmark_dict = {
        # latency [cycle]: reported latency per iteration, energy [nJ]: reported energy per iteration,
        # latency_model [cycle]: latency to be modeled, energy_model [nJ]: energy to be modeled
        "TSP_1K": {
            "num_spins": 1000,
            "num_js": 999 * 1000,
            "num_iterations": 1,
            "w_pres": 4,
            "latency": 80,
            "energy": 5000,
            "latency_model": 0,
            "energy_model": 0,
        },  # note the num_js is not halved as SACHI stores J twice
        "MD_1K": {
            "num_spins": 1000,
            "num_js": 8 * 1000,
            "num_iterations": 1,
            "w_pres": 4,
            "latency": 80,
            "energy": 40,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MD_500": {
            "num_spins": 500,
            "num_js": 8 * 500,
            "num_iterations": 1,
            "w_pres": 2,
            "latency": 80,
            "energy": 10.5,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MD_100K": {
            "num_spins": 100 * 1000,
            "num_js": 8 * 100 * 1000,
            "num_iterations": 1,
            "w_pres": 2,
            "latency": 12500,
            "energy": 2000,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MD_200K": {
            "num_spins": 200 * 1000,
            "num_js": 8 * 200 * 1000,
            "num_iterations": 1,
            "w_pres": 2,
            "latency": 25000,
            "energy": 4000,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MD_300K": {
            "num_spins": 300 * 1000,
            "num_js": 8 * 300 * 1000,
            "num_iterations": 1,
            "w_pres": 2,
            "latency": 37500,
            "energy": 6000,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MD_1M": {
            "num_spins": 1000 * 1000,
            "num_js": 8 * 1000 * 1000,
            "num_iterations": 1,
            "w_pres": 2,
            "latency": 125000,
            "energy": 20000,
            "latency_model": 0,
            "energy_model": 0,
        },
    }

    # calculating the performance metrics
    for benchmark, info in benchmark_dict.items():
        num_spins = info["num_spins"]
        num_js = info["num_js"]
        num_iterations = info["num_iterations"]
        w_pres = info["w_pres"]
        energy = info["energy"]
        latency = benchmark_dict[benchmark]["latency"]
        # adding additional modeling setting, when the problem size exceeds the compute memory size
        # (num_spins > num_cores * compute_memory_depth)
        if num_spins > num_cores * compute_memory_depth:
            parallelism = (
                num_cores / 2
            )  # half of the cores are used, I suppose the reason is they need to store the spins twice in the memory
        else:
            parallelism = 0  # not used
        # calculating the energy
        energy_model = (
            energy_per_spin_per_degree_per_bit * w_pres * num_js * num_iterations / 1000
        )  # pJ -> nJ
        # calculating the latency
        latency_model = (
            compute_memory_depth if parallelism == 0 else num_spins / parallelism
        )
        logging.info(
            f"Benchmark: {benchmark}, Latency (model): {latency_model} cycles, Latency (reported): {latency} cycles, "
            f"Energy (model): {energy_model} nJ, Energy (reported): {energy} nJ"
        )
        benchmark_dict[benchmark]["energy_model"] = energy_model
        benchmark_dict[benchmark]["latency_model"] = latency_model
    return benchmark_dict


if __name__ == "__main__":
    """
    validating the modeling results to SACHI (HPCA'24)
    """
    logging_level = logging.INFO  # logging level
    logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=logging_level, format=logging_format)
    plot_results_in_bar_chart(
        validation_to_sachi(), output_file="output/sachi.png", text_type="absolute"
    )
