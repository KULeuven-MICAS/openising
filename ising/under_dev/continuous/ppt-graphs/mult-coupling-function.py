from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-1.5, 1.5, 1001)
y = x
print(True)

# Configure plots
plt.style.use('petroff10')
fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
print(True)

ax.plot(x, y, lw=2)
ax.set_xlabel(r"$x_j$")
ax.set_ylabel(r"$k(x_j)$")
ax.axhline(0, color="black", ls="--", lw=1)
ax.axvline(0, color="black", ls="--", lw=1)
ax.axvline(1, color="black", ls=":", lw=1)
ax.axvline(-1, color="black", ls=":", lw=1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
print(True)

# Save and plot
DIR = Path('/mnt/c/Users/tbettens/OneDrive - KU Leuven/Ising/presentations/Continuous Ising Machines')
SAVE_PATH = DIR / (Path(__file__).stem + ".png")
plt.savefig(SAVE_PATH)
plt.show()

