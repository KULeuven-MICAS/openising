import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from ising.flow import TOP
figtop = TOP / "ising/under_dev/Flipping/Figures"


def plot_data(data, figname, xlabel:str, xticks:list[str], ylabel:str, yticks:list[str], best_found:float, colorbar_label:str= "Energy"):
    plt.figure()
    ax = plt.gca()
    plt.imshow(data, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(len(yticks)), yticks)
    plt.colorbar(label=colorbar_label, ticks=np.arange(best_found, np.max(data)+1, 50, dtype=int)) 
    plt.savefig(figtop / figname, bbox_inches="tight")
    plt.close()

def make_bar_plot(dataframes:dict[str:pd.DataFrame], xaxis_name, yaxis_name, figname, best_found:float):
    # concat the dataframes
    plt.figure()
    if len(dataframes.keys()) > 1:
        data = dict()
        for df_name in dataframes.keys():
            data[df_name] = dataframes[df_name].melt()
        df = pd.concat(data, names=["source", "old_index"])
        df = df.reset_index(level=0).reset_index(drop=True)
        sns.boxplot(data=df, x='variable', y='value', hue="source")
    else:
        df = dataframes[list(dataframes.keys())[0]]
        sns.boxplot(data=df, x=xaxis_name, y=yaxis_name)
    plt.axhline(y=best_found, color='k', linestyle='--', label='Best found')
    plt.legend()
    plt.xticks(ticks=range(100), rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.savefig(figtop / figname, bbox_inches="tight")
    plt.close()