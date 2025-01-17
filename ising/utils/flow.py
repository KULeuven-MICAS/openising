import pathlib

from ising.utils.helper_solvers import return_rx
from ising.utils.HDF5Logger import return_metadata

def make_directory(path:pathlib.Path) -> None:
    """Makes the given directory if it does not exist.

    Args:
        path (pathlib.Path): the directory to create
    """
    path.mkdir(parents=True, exist_ok=True)

def parse_hyperparameters(args, num_iter) -> dict[str:float]:
    hyperparameters = dict()
    # BRIM parameters
    tend = float(args.t_end)
    hyperparameters["dtBRIM"] = tend / num_iter
    hyperparameters["C"] = float(args.C)
    hyperparameters["G"] = float(args.G)

    hyperparameters["kmin"] = float(args.k_min)
    hyperparameters["kmax"] = float(args.k_max)
    hyperparameters["flip"] = bool(args.flip)

    # SA parameters
    hyperparameters["T"] = float(args.T)
    Tfin = float(args.T_final)
    hyperparameters["r_T"] = return_rx(num_iter, hyperparameters["T"], Tfin)

    # SCA parameters
    hyperparameters["q"] = float(args.q)

    # SB parameters
    hyperparameters["dtSB"] = float(args.dt)
    hyperparameters["a0"] = float(args.a0)
    hyperparameters["c0"] = float(args.c0)
    hyperparameters["seed"] = int(args.seed)

    return hyperparameters

def get_best_found_gurobi(gurobi_files:list[pathlib.Path]) -> list[float]:
    best_found_list = []
    for file in gurobi_files:
        best_found = return_metadata(file, "solution_energy")
        best_found_list.append(best_found)
    return best_found_list
