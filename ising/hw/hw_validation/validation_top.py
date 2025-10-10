import logging
import matplotlib.pyplot as plt
import os
from api import plot_results_in_bar_chart, plot_results_in_bar_chart_prim_caefa
from sachi import validation_to_sachi
from prim_caefa import plot_results_in_bar_chart_with_breakdown
from fpga_asb_v2 import validation_to_fpga_asb_v2
from fpga_asb import validation_to_fpga_asb

if __name__ == "__main__":
    """
    This script is used to validate the hardware performance model with the reported performance
    TODO: add bar breakdown to the validation results
    TODO: keep the validation plot tick name to be consistent with each other
    """
    logging_level = logging.INFO  # logging level
    logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=logging_level, format=logging_format)

    validation_list = ["prim_caefa", "sachi", "fpga_asb_v2", "fpga_asb_v1"]

    if os.path.exists("output") is False:
        os.makedirs("output")

    for validation in validation_list:
        if validation == "sachi":
            benchmark_dict = validation_to_sachi()
            plot_results_in_bar_chart(
                benchmark_dict, output_file="output/sachi.svg", text_type="absolute"
            )
        elif validation == "prim_caefa":
            benchmark_dict = plot_results_in_bar_chart_with_breakdown()
            plot_results_in_bar_chart_prim_caefa(
                benchmark_dict,
                output_file="output/prim_caefa.svg",
                text_type="relative",
                with_latency_breakdown=True,
                latency_normalize=True
            )
        elif validation == "fpga_asb_v2":
            benchmark_dict = validation_to_fpga_asb_v2()
            plot_results_in_bar_chart_prim_caefa(
                benchmark_dict,
                output_file="output/fpga_asb_v2.svg",
                text_type="absolute",
                with_latency_breakdown=True,
                latency_normalize=False
            )
        elif validation == "fpga_asb_v1":
            benchmark_dict = validation_to_fpga_asb()
            plot_results_in_bar_chart_prim_caefa(
                benchmark_dict,
                output_file="output/fpga_asb_v1.svg",
                text_type="absolute",
                with_latency_breakdown=True,
                latency_normalize=False
            )
        else:
            raise ValueError(f"Unknown validation method: {validation}")