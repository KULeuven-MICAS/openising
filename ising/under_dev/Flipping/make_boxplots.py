import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import make_dataclass

from ising.flow import TOP
file = TOP / "ising/under_dev/Flipping/9M_results"
save_path = TOP / "ising/under_dev/Flipping/9M_results"
orig = TOP / "ising/flow/no_backup/9M_results"

nb_trials = 100
nb_appl = 3

benchmarks = {"TSP": "burma14", "QKP":"jeu_100_25_1", "MIMO":"MIMO"}
best_energies = { "TSP": 3323, "QKP":-18558, "MIMO": 0.}
Result = make_dataclass("Result", [("technique", str), ("energies", pd.Series)])

# plt.figure()
# fig, axes = plt.subplots(1, nb_trials)

df = pd.DataFrame(columns=["application", "technique", "energy"])

for app, benchmark in benchmarks.items():
    # ax = plt.subplots(1, nb_trials, i+1)
    file_orig = orig / f"{app}/{app}_Mult_energies.csv"
    file_rand = file / f"{benchmark}_energies_random_flipping.pkl"
    file_grad = file / f"{benchmark}_energies_gradient_flipping.pkl"
    file_freq = file / f"{benchmark}_energies_frequency_flipping.pkl"

    energies_orig = np.loadtxt(file_orig)
    energies_rand = np.loadtxt(file_rand)
    energies_grad = np.loadtxt(file_grad)
    energies_freq = np.loadtxt(file_freq)
    dforig = pd.DataFrame({"application": app, "technique":"no flipping", "energy": energies_orig})
    dfrand = pd.DataFrame({"application":app, "technique":"random", "energy": energies_rand})
    dfgrad = pd.DataFrame({"application":app, "technique":"gradient", "energy": energies_grad})
    dffreq = pd.DataFrame({"application":app, "technique":"frequency", "energy": energies_freq})
    df = pd.concat([df, dforig, dfrand, dfgrad, dffreq])
    # sns.boxplot(pd.DataFrame({"bSB":energies_bSB, "Ours":energies_Mult}), orient='v', ax=axes[i])
    # axes[i].set_title(f"Results for {app}")
    # axes[i].axhline(best_energies[app], color='k', linestyle="--", label="Best found")
    # axes[i].legend()
# plt.savefig(save_path / "boxplot_energies.png")
# plt.close()

fig, axes = plt.subplots(nb_appl, 1, sharex=True)
fig.set_size_inches(11, 13)
for name, ax in zip(benchmarks.keys(), axes.flatten()):
    sns.boxplot(data=df[df["application"] == name], x="technique", y="energy", ax=ax, palette="tab10")
    ax.axhline(best_energies[name], color='k', linestyle="--", label=f"Best found: {best_energies[name]}")
    ax.set_title(f"Results for {name}")
    ax.legend(loc="center", bbox_to_anchor=(1.12, .5))
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.savefig(save_path / "boxplot_energies_orig_flipping_techniques.png", bbox_inches="tight")
plt.close()
