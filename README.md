# 🧮**OpenIsing**

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## **Codebase Structure**

The detailed structure of this repository is documented in our
[Notion page](https://feather-broom-8b3.notion.site/Codebase-Structure-99120b5f9c57424fa1ef008c94dab172).
Please refer to this resource for a complete breakdown of the directories and their purposes.

## **Getting Started**

### **Requirements**
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.
- **Cadance Spectre**: Any version of Spectre will do. Only required if you want to run analog circuit simulations.

### **Setup**
 
```bash
git clone git@gitlab.esat.kuleuven.be:ising-project/ising.git
cd ising
source .setup
```

## **How to get results**
To simulate, just run:
```bash
python main.py
```

The readme for the configuration can be found in the [readme](./ising/inputs/config/README.md) of the folder.

It is allowed to use [Gurobi](https://www.gurobi.com/), indicated by the argument `-use_gurobi`. However, it can only be used when you have an active [Gurobi license](https://www.gurobi.com/solutions/licensing/).
