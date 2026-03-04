# 🧮**OpenIsing**

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## **Getting Started**

### **Requirements**
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.

### **Setup**
 
```bash
git clone git@github.com:KULeuven-MICAS/openising.git
cd openising
source .setup
```

## **How to get results**
To simulate, just run:
```bash
python main.py
```

A configuration file (YAML) is required as the input for the framework. The readme for the configuration can be found in the [readme](./ising/inputs/config/README.md) of the folder. 

For testing multiple values of the same parameter, you can run:
```bash
python main_loop.py
```

This simulation will run for the given problems and parameter values. For each correpsonding problem a histogram and boxplot are generated and stored under `ising/outputs/<problem>/plots`.

It is allowed to use [Gurobi](https://www.gurobi.com/), indicated by the argument `-use_gurobi`. However, it can only be used when you have an active [Gurobi license](https://www.gurobi.com/solutions/licensing/).
