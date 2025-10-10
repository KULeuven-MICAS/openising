import matplotlib.pyplot as plt
import logging


def plot_results_in_bar_chart(
    benchmark_dict: dict, output_file: str = "sachi.png", text_type: str = "absolute"
    ):
    """
    plot the modeling results in bar chart
    :param benchmark_dict: dict
    :param output_file: otuput png file to save as
    :param text_type: text type to annotate, either absolute or relative
    """
    assert text_type in ["absolute", "relative"]
    colors = [
        # "#cd87de",
        # "#fff6d5",
        # "#5df3ce",
        # "#2ca02c",
        # "#d62728",
        # "#9467bd",
        "#8c564b",
        # "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    benchmark_names = list(benchmark_dict.keys())
    energy_model = [
        benchmark_dict[benchmark]["energy_model"] for benchmark in benchmark_names
    ]
    energy_reported = [
        benchmark_dict[benchmark]["energy"] for benchmark in benchmark_names
    ]
    latency_model = [
        benchmark_dict[benchmark]["latency_model"] for benchmark in benchmark_names
    ]
    latency_reported = [
        benchmark_dict[benchmark]["latency"] for benchmark in benchmark_names
    ]
    x = list(range(len(benchmark_names)))
    width = 0.35
    ax[0].bar(
    x, latency_model, width, label="Compute", color=colors[0], edgecolor="black"
    )
    ax[0].bar(
        [i + width for i in x],
        latency_reported,
        width,
        label="Reported",
        color=colors[1],
        edgecolor="black",
    )
    ax[0].set_ylabel("Latency [cycles]", weight="bold")
    # ax[0].set_title("Latency Validation")
    ax[0].set_xticks([i + width / 2 for i in x])
    ax[0].set_xticklabels(benchmark_names)
    ax[0].legend()
    ax[1].bar(x, energy_model, width, label="Compute", color=colors[0], edgecolor="black")
    ax[1].bar(
        [i + width for i in x],
        energy_reported,
        width,
        label="Reported",
        color=colors[1],
        edgecolor="black",
    )
    ax[1].set_ylabel("Energy [nJ]", weight="bold")
    # ax[1].set_title("Energy Validation")
    ax[1].set_xticks([i + width / 2 for i in x])
    ax[1].set_xticklabels(benchmark_names)
    ax[1].legend()
    if text_type == "absolute":
        # add the text labels to show the absolute values
        for i in range(len(benchmark_names)):
            ax[0].text(
                i + width / 2,
                latency_model[i] + 0.1,
                f"{int(latency_model[i])}",
                ha="center",
                va="bottom",
                weight="bold",
            )
            ax[1].text(
                i + width / 2,
                energy_model[i] + 0.1,
                f"{int(energy_model[i])}",
                ha="center",
                va="bottom",
                weight="bold",
            )
    else:
        # add the text labels to show the modeling mismatch
        for i in range(len(benchmark_names)):
            if energy_reported[i] != 0:
                energy_mismatch = int(
                    (energy_model[i] / energy_reported[i] - 1) * 100
                )  # in percentage
                ax[1].text(
                    i + width / 2,
                    energy_model[i] + 0.1,
                    f"{energy_mismatch}%",
                    ha="center",
                    va="bottom",
                    weight="bold",
                )
            latency_mismatch = int(
                (latency_model[i] / latency_reported[i] - 1) * 100
            )  # in percentage
            ax[0].text(
                i + width / 2,
                latency_model[i] + 0.1,
                f"{latency_mismatch}%",
                ha="center",
                va="bottom",
                weight="bold",
            )

    # set the y scale to log scale
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    ax[0].grid()
    ax[1].grid()
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info(f"Figure (bar chart) saved to {output_file}")
    pass

def plot_results_in_bar_chart_with_breakdown(
    benchmark_dict: dict, output_file: str = "sachi.png", text_type: str = "absolute",
    with_latency_breakdown: bool = False,
    latency_normalize: bool = True
):
    """
    plot the modeling results in bar chart
    :param benchmark_dict: dict
    :param output_file: otuput png file to save as
    :param text_type: text type to annotate, either absolute or relative
    :param with_latency_breakdown: whether to include latency breakdown in the bar chart
    :param latency_normalize: whether to normalize the latency to the minimum reported value
    """
    assert text_type in ["absolute", "relative"]
    colors = [
        # "#cd87de",
        # "#fff6d5",
        # "#5df3ce",
        # "#2ca02c",
        # "#d62728",
        # "#9467bd",
        "#8c564b",
        # "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    # plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    benchmark_names = list(benchmark_dict.keys())
    energy_model = [
        benchmark_dict[benchmark]["energy_model"] for benchmark in benchmark_names
    ]
    energy_reported = [
        benchmark_dict[benchmark]["energy"] for benchmark in benchmark_names
    ]
    latency_model = [
        benchmark_dict[benchmark]["latency_model"] for benchmark in benchmark_names
    ]
    latency_reported = [
        benchmark_dict[benchmark]["latency"] for benchmark in benchmark_names
    ]
    min_latency_reported = min(latency_reported)

    if latency_normalize:
        # normalize the latency and energy to the minimum reported value
        latency_model = [latency / min_latency_reported for latency in latency_model]
        latency_reported = [latency / min_latency_reported for latency in latency_reported]

    x = list(range(len(benchmark_names)))
    width = 0.35
    if with_latency_breakdown:
        latency_breakdown: list[dict] = [
            benchmark_dict[benchmark].get("latency_breakdown", None)
            for benchmark in benchmark_names
        ]
        latency_breakdown_sram = [
            latency_breakdown[i]["sram"] if latency_breakdown[i] is not None else 0
            for i in range(len(benchmark_names))
        ]
        latency_breakdown_spin_update = [
            latency_breakdown[i]["spin update"]
            if latency_breakdown[i] is not None
            else 0
            for i in range(len(benchmark_names))
        ]

        if latency_normalize:
            # normalize the breakdown to the minimum reported value
            latency_breakdown_sram = [latency / min_latency_reported for latency in latency_breakdown_sram]
            latency_breakdown_spin_update = [latency / min_latency_reported for latency in latency_breakdown_spin_update]

        ax[0].bar(x, latency_breakdown_spin_update, width, label="Compute", color=colors[0], edgecolor="black")
        ax[0].bar(x, latency_breakdown_sram, width, bottom=latency_breakdown_spin_update, label="Weight Loading", color=colors[3], edgecolor="black")
    else:
        ax[0].bar(
        x, latency_model, width, label="Compute", color=colors[0], edgecolor="black"
        )
    ax[0].bar(
        [i + width for i in x],
        latency_reported,
        width,
        label="Reported",
        color=colors[1],
        edgecolor="black",
    )
    if latency_normalize:
        ax[0].set_ylabel("Latency [Normalized]", weight="bold")
    else:
        ax[0].set_ylabel("Latency [cycles]", weight="bold")
    # ax[0].set_title("Latency Validation")
    ax[0].set_xticks([i + width / 2 for i in x])
    ax[0].set_xticklabels(benchmark_names)
    ax[0].legend(loc="lower right")
    ax[1].bar(x, energy_model, width, label="Compute", color=colors[0], edgecolor="black")
    ax[1].bar(
        [i + width for i in x],
        energy_reported,
        width,
        label="Reported",
        color=colors[1],
        edgecolor="black",
    )
    if latency_normalize:
        ax[1].set_ylabel("Energy [Normalized]", weight="bold")
    else:
        ax[1].set_ylabel("Energy [nJ]", weight="bold")
    # ax[1].set_title("Energy Validation")
    ax[1].set_xticks([i + width / 2 for i in x])
    ax[1].set_xticklabels(benchmark_names)
    ax[1].legend(loc="lower right")
    if text_type == "absolute":
        # add the text labels to show the absolute values
        for i in range(len(benchmark_names)):
            ax[0].text(
                i + width / 2,
                latency_model[i] + 0.1,
                f"{int(latency_model[i])}",
                ha="center",
                va="bottom",
                weight="bold",
            )
            ax[1].text(
                i + width / 2,
                energy_model[i] + 0.1,
                f"{int(energy_model[i])}",
                ha="center",
                va="bottom",
                weight="bold",
            )
    else:
        # add the text labels to show the modeling mismatch
        for i in range(len(benchmark_names)):
            if energy_reported[i] != 0:
                energy_mismatch = int(
                    (energy_model[i] / energy_reported[i] - 1) * 100
                )  # in percentage
                ax[1].text(
                    i + width / 2,
                    energy_model[i] + 0.1,
                    f"{energy_mismatch}%",
                    ha="center",
                    va="bottom",
                    weight="bold",
                )
            latency_mismatch = int(
                (latency_model[i] / latency_reported[i] - 1) * 100
            )  # in percentage
            ax[0].text(
                i + width / 2,
                latency_model[i] + 0.05,
                f"{latency_mismatch}%",
                ha="center",
                va="bottom",
                weight="bold",
            )

    # set the y scale to log scale
    # ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    # rotate the x ticklabels
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add grid and put grid below axis
    ax[0].grid()
    ax[1].grid()
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    plt.tight_layout()
    if output_file.endswith(".svg"):
        plt.savefig(output_file, format="svg")
    else:
        plt.savefig(output_file)
    logging.info(f"Figure (bar chart) saved to {output_file}")
    pass
