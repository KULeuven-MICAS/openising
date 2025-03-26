from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Generate data
delta = np.linspace(-np.pi, np.pi, 1001)
def K(delta):
    return np.cos(delta)

x = np.linspace(-1.2*np.pi/2, 1.2*np.pi/2, 1001)
def D(x, k=2):
    return -np.cos(2*x)

# Configure plots
plt.style.use('petroff10')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), constrained_layout=True)

ax1.plot(delta, K(delta), lw=2)
ax1.axvline(0, color="black", ls="--", lw=1)
ax1.set_xlabel(r"$\Delta_{ij}$")
ax1.set_ylabel(r"$K(\Delta_{ij})$")
ax1.set_yticks([])

ax2.plot(x, D(x), lw=2)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.axvline(-np.pi/2, color="black", ls="--", lw=1)
ax2.axvline(0, color="black", ls="--", lw=1)
ax2.axvline(np.pi/2, color="black", ls="--", lw=1)
ax2.set_xlabel(r"$x_i$")
ax2.set_ylabel(r"$D(x_i)$")
ax2.set_yticks([])
ax2.set_xticks([-np.pi/2, 0, np.pi/2], labels=[r"$x_{-}$", r"$x_{m}$", r"$x_{+}$"], va="center")

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + ".png")
plt.savefig(SAVE_PATH)
plt.show()


