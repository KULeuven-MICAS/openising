import matplotlib.pyplot as plt
import numpy as np

from ising.stages.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.stages import TOP


def plot_model(model: IsingModel, figName: str, output_path: str="ising/outputs/model"):
    coupling = triu_to_symm(model.J)
    density = np.count_nonzero(coupling) / coupling.size
    if not (TOP / "ising/outputs/model").exists():
        (TOP / "ising/outputs/model").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.spy(coupling, markersize=1)
    ax.set(
        title=f"Matrix sparsity pattern — {density:.1%} nonzero",
        xlabel="Column",
        ylabel="Row",
    )
    fig.savefig(TOP / f"{output_path}/{figName}.pdf")
    plt.close()

    plt.figure()
    im = plt.imshow(
        coupling,
        cmap="PRGn",
        vmin=-np.max(np.abs(coupling)),
        vmax=np.max(np.abs(coupling)),
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(im, label="Coupling strength")
    plt.xlabel('Column')
    plt.ylabel("Row")
    plt.title("Coupling matrix")
    plt.tight_layout()
    plt.savefig(TOP / f"{output_path}/{figName}_heatmap.pdf")
    plt.close()

def plot_model_distribution(model: IsingModel, figName: str, output_path:str="ising/outputs/model", bins: int = 50):
    """Plots the distribution of the given model as a histrogram plot.

    @type model: IsingModel
    @param model: the model that will be plotted
    @type figName: str
    @param figName: the name of the figure
    @type bins: int, optional
    @param bins: The amount of bins for grouping the histogram bars. Defaults to 50.
    """
    coupling = triu_to_symm(model.J)
    bias = model.h
    if not (TOP / "ising/outputs/model").exists():
        (TOP / "ising/outputs/model").mkdir(parents=True, exist_ok=True)
    _, axes = plt.subplots(2, 1)
    for ax, data in zip(axes, (coupling, bias)):
        ax.hist(data.reshape(-1, 1), bins=bins, alpha=0.7, color="blue", edgecolor="black")
        plt.title(
            f"Distribution of values (min: {round(np.min(data), 2):.2f}, "
            f"max: {round(np.max(data), 2):.2f}, mean: {round(np.mean(data), 2):.2f}, "
            f"unique levels: {len(np.unique(data))})",
            fontsize=10,
            weight="bold",
        )
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(TOP / f"{output_path}/{figName}.pdf", bbox_inches="tight")
    plt.close()
