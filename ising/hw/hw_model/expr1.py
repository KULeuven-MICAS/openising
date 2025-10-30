import matplotlib.pyplot as plt
import numpy as np

"""
This is a script to prove that spin updating in sachi takes significant latency and energy.
We create two subfigures:
1. Left: Stacked bar chart showing latency breakdown for PEs and CIM
2. Right: Pie chart showing energy distribution for CIM
"""

# Create figure with 2 subplots (left and right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))

# ===== LEFT SUBFIGURE: Stacked Bar Chart =====
# Data for stacked bars
bar_labels = ['PEs', 'CIM']
data_processing = [80, 80]
spin_updating = [0, 160]

colors = ['#FF6B6B',  # DRAM (Off-chip Memory)
          '#4ECDC4',  # L1 (On-chip Memory)
          '#F7DC6F',  # SU (Spin Updating)
          '#45B7D1',  # MAC (MACs)
          '#FFA07A',  # ADD (Adds)
          '#98D8C8']  # COMP (COMPs)

# X-axis positions
x_pos = np.arange(len(bar_labels))

# Create stacked vertical bars
bars1 = ax1.bar(x_pos, data_processing, label='Data Processing', 
                color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x_pos, spin_updating, bottom=data_processing, 
                label='Spin Updating (SU)', color='#F7DC6F', alpha=0.8, 
                edgecolor='black', linewidth=1.2)

# Customize left plot
ax1.set_ylabel('Latency [cycles]', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(bar_labels, fontsize=11)
# ax1.legend(loc='upper left')
# ax1.grid(axis='y', alpha=0.3, linestyle='--')
# ax1.set_axisbelow(True)

# Add star
ax1.text(
    1,
    data_processing[1] + spin_updating[1],
    "\u2605",
    ha="center",
    va="center",
    weight="bold",
    fontsize=15,
)

# Add value labels on top of bars
sizes = [data_processing[i] + spin_updating[i] for i in range(len(bar_labels))]
for i in range(len(bar_labels)):
    ax1.text(
                i,
                sizes[i] * 1.05,
                f"{sizes[i]}",
                ha="center",
                va="bottom",
                weight="bold",
            )

ax1.set_ylim(0, 300)

# ===== RIGHT SUBFIGURE: Pie Chart =====
# Data for pie chart
pie_labels = ['PEs', 'CIM-read', 'CIM-SU']
pie_data = [90.72, 729, 4030.464]

# Colors for pie chart
pie_colors = ['#FF6B6B', '#4ECDC4', '#F7DC6F']

# Create pie chart
wedges, texts, autotexts = ax2.pie(
    pie_data,
    labels=pie_labels,
    autopct='%1.f%%',
    colors=pie_colors,
    startangle=90,
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
    textprops={'fontsize': 11}
)

# Make percentage text bold
# for autotext in autotexts:
#     autotext.set_color('white')
#     autotext.set_fontweight('bold')
#     autotext.set_fontsize(10)

# Customize right plot
# ax2.set_title('Energy Distribution', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.savefig('output/latency_energy_breakdown.png', dpi=300)