from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-2, 2, 1001)
y = 1 - 1/2 * x**2

# Configure plots
plt.style.use('petroff10')
fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)

ax.plot(x, y, lw=2)
ax.set_xlabel(r"$\Delta_{ij}$")
ax.set_ylabel(r"$K(\Delta_{ij}$")
ax.axhline(0, color="black", ls="--", lw=1)
ax.axvline(0, color="black", ls="--", lw=1)

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + ".png")
plt.savefig(SAVE_PATH)
plt.show()
