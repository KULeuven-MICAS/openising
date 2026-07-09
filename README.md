# 🧮OpenIsing

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## 🚀 Installation

### Requirements
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.
Install this git repository by running
```bash
git clone git@github.com:KULeuven-MICAS/openising.git
```
### Linux Setup

The best way to set up the environment is through either a virtual environment, or a conda virtual environment. I recommend first creating a virtual environment and then setting this as a default in VSCode.

**1. Virtual Environment**

Follow the following script to set up using a virtual environment.
```bash
cd openising
python -m venv /path/to/new/virtual/environment
source .setup
```
The last step will activate the virtual environment and install all dependencies. Therefore, you will need to run this setup file every time you start a new terminal. However, with VSCode you can make sure this happens automatically.

**2. Conda Environment**

Be sure conda is available. Test this by running:
```bash
conda list
```
If you get the message that conda doesn't exist, follow the instruction on this [page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Now we can make a new conda environment. Run
```bash
conda create --name <my-env> python=3.12.7
```
where `<my-env>` is the name of your environment. Now everytime you open a terminal you have to activate the environment by running

```bash
conda activate <my-env>
```
However, with VSCode you can make sure this happens automatically.

**VSCode Default Setting**

If you are using VSCode the virtual environment can also be created and set as a default. But when you have already made the environment from one of the choices above, you're already halfway there. All information is available on this [page](https://code.visualstudio.com/docs/python/environments).
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