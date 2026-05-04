# 🧮OpenIsing

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## 🚀 Installation

### Requirements
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.

### Linux Setup

```bash
git clone git@github.com:KULeuven-MICAS/openising.git
cd openising
source .setup
```
### Windows Setup
TODO

## 📉 How to get results
We have two examples showcasing how results can be gathered from the framework. The first example runs the bSB solver on a dummy Max Cut problem. Run the following command to get results:
```bash
python main.py
```
If you want to change the problem to e.g. TSP, change the ``problem_type`` parameter in the file ``main.py``. A configuration file (YAML) is required as the input for the framework, which can be changed in the ``config_path`` parameter. A thorough explanation of each parameter required for the configuration can be found in the [readme](./ising/inputs/config/README.md) of the folder. 

For testing multiple values of the same parameter, you can run:
```bash
python main_loop.py
```

This simulation will run for the given problems and parameter values. For each correpsonding problem a histogram and boxplot are generated and stored under `ising/outputs/<problem>/plots`.

It is allowed to use [Gurobi](https://www.gurobi.com/), indicated by the argument `-use_gurobi`. However, it can only be used when you have an active [Gurobi license](https://www.gurobi.com/solutions/licensing/).

# 💻 Contributing

We welcome contributions! Feel free to fork the repository, submit PRs, or open issues.