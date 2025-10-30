import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['DRAM', 'L1', 'SU', 'MAC', 'ADD', 'COMP']
energy = [3.7, 0.05, 0.013, 0.0028, 2 * 0.567e-3, 0.567e-3]

# Colors from the first op breakdown bar chart
colors = ['#FF6B6B',  # DRAM (Off-chip Memory)
          '#4ECDC4',  # L1 (On-chip Memory)
          '#F7DC6F',  # SU (Spin Updating)
          '#45B7D1',  # MAC (MACs)
          '#FFA07A',  # ADD (Adds)
          '#98D8C8']  # COMP (COMPs)

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 3))

# X-axis positions
x_pos = np.arange(len(labels))

# Create vertical bar chart
# For horizontal bar chart, change to: ax.barh(x_pos, energy)
bars = ax.bar(x_pos, energy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Customize the plot
ax.set_xlabel('Component', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy [pJ/bit]', fontsize=12, fontweight='bold')
# ax.set_title('Energy Consumption by Component', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.set_yscale('log')

# Optional: Add value labels on top of bars
# for i, (bar, val) in enumerate(zip(bars, energy)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height,
#             f'{val:.2e}' if val < 0.01 else f'{val:.3f}',
#             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('energy_component_breakdown.png', dpi=300)
