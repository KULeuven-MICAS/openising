from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Generate data
x = np.linspace(-2, 2, 1001)
def binarisation(x, k=2):
    return np.tanh(k * np.tanh(k*x)) - x
y = [quad(binarisation, 0, x)[0] for x in x]

# Configure plots
plt.style.use('petroff10')
fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)

ax.plot(x, y, lw=2)
ax.axvline(1, color="black", ls="--", lw=1)
ax.axvline(0, color="black", ls="--", lw=1)
ax.axvline(-1, color="black", ls="--", lw=1)
ax.set_yticks([])
ax.set_xticks([-1, 0, 1], labels=[r"$x_{-}$", r"$x_{m}$", r"$x_{+}$"])
ax.set_ylabel(r"$D(x_i)$")

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + ".png")
plt.savefig(SAVE_PATH)
plt.show()

