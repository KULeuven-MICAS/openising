import pathlib

def get_optim_value(benchmark:str, optim_file: pathlib.Path):
    """Returns the best found energy of the given benchmark.

    Args:
        benchmark (pathlib.Path): the given benchmark.
        optim_file (pathlib.Path): the path to the file containing the optimal energies.
    """
    benchmark = str(benchmark).split("/")[-1].split(".")[0]
    best_found = None

    with optim_file.open() as f:
        for line in f:
            line = line.split()
            if line[0] == benchmark:
                best_found = float(line[1])
                break

    return best_found
