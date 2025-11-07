import matplotlib.pyplot as plt
import numpy as np


############################################
## Parameters
N = 100
D = N
T = 10
B_j = 8
B_p = 16
############################################

# Sample data for 4 bars, each with 6 segments
# Replace these values with your actual data
bar_labels = ['Unspecific Weight\n(exceeds L1)', 'Specific Weight\n(exceeds L1)', 'Unspecific Weight\n(fits L1)', 'Specific Weight\n(fits L1)']

# Data for each segment (values for each of the 4 bars)
off_chip_memory = [N * (D+1) * T * B_j, N * (D+1) * T * B_j, N * B_j, N * (D+1) * B_j]
on_chip_memory = [N * (D+1) * T * B_j] * 4
macs = [N * D * T * B_j] * 4
adds = [N * T * B_p] * 4
comps = [N * T * B_p] * 4
spin_updating = [2 * N * D * T] * 4

# Create the plot
fig, ax = plt.subplots(figsize=(5, 3))

# Y-axis positions
y_pos = np.arange(len(bar_labels))

# Create stacked horizontal bars
bars1 = ax.barh(y_pos, off_chip_memory, label='DRAM', color='#FF6B6B', edgecolor='black')
bars2 = ax.barh(y_pos, on_chip_memory, left=off_chip_memory, label='L1', color='#4ECDC4', edgecolor='black')
# bars3 = ax.barh(y_pos, spin_updating,
#                 left=np.array(off_chip_memory) + np.array(on_chip_memory),
#                 label='CIM memory', color='#F7DC6F', edgecolor='black')
bars4 = ax.barh(y_pos, adds,
                left=np.array(off_chip_memory) + np.array(on_chip_memory),
                label='Adder', color='#FFA07A', edgecolor='black')
bars5 = ax.barh(y_pos, comps,
                left=np.array(off_chip_memory) + np.array(on_chip_memory) + np.array(adds),
                label='Comparator', color='#98D8C8', edgecolor='black')
bars6 = ax.barh(y_pos, macs,
                left=np.array(off_chip_memory) + np.array(on_chip_memory) + np.array(adds) + np.array(comps),
                label='MAC', color='#45B7D1', edgecolor='black')

# Customize the plot
# ax.set_ylabel('Models', fontsize=12)
ax.set_xlabel('#operators (bit-wise)', fontsize=12)
# ax.set_title('Breakdown of Bit-wise Operations', fontsize=14, fontweight='bold')
ax.set_yticks(y_pos)
ax.set_yticklabels(bar_labels)
ax.legend(loc='upper right')

# Add grid for better readability
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('op_component_breakdown.png', dpi=300)
