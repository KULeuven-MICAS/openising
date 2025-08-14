import pandas as pd #type: ignore

from ising.flow import TOP
from ising.under_dev.Flipping.new_strategy import make_bar_plot

file = TOP / "ising/under_dev/Flipping/exponential_change.out"
init_sizes = ["0.5","0.66", "0.75", "0.8", "0.83", "0.875", "0.9", "1.0"]
final_sizes = ["1/20", "1/18", "1/16", "1/14", "1/12", "1/10", "1/8", "1/6"]
model_size = 196
nb_flip = 100
nb_runs = 10

energies = {f"initial size {init_size}": {f"final size {final_size}": {iteration: [] for iteration in range(nb_flip)} for final_size in final_sizes} for init_size in init_sizes}
current_init_size = 0
current_final_size = 0
iteration = 0
run = 0
first_line = True
with open(file, "r") as f:
    for line in f:
        if first_line:
            first_line = False
            continue
        line_split = line.split()
        if line_split[0] == "INFO:Initial":
            iteration = 0
            # Case when run hasn't ended
            if run < nb_runs:
                run += 1
            # Case when next initial size is chosen
            elif current_final_size == len(final_sizes) - 1:
                current_init_size += 1
                current_final_size = 0
                run = 0
            # Case when next final size is chosen
            else:
                current_final_size += 1
                run = 0
        elif line_split[0] == "INFO:Done":
            energy = float(line_split[-1])
            init_size = init_sizes[current_init_size]
            final_size = final_sizes[current_final_size]
            energies[f"initial size {init_size}"][f"final size {final_size}"][iteration].append(energy)
            iteration += 1

for init_size in init_sizes:
    for final_size in final_sizes:
        energies[f"initial size {init_size}"][f"final size {final_size}"] = pd.DataFrame(energies[f"initial size {init_size}"][f"final size {final_size}"])
    make_bar_plot(energies[f"initial size {init_size}"], "Iteration", "Energy", f"exponential_change_{int(float(init_size)*model_size)}.png", 3323)