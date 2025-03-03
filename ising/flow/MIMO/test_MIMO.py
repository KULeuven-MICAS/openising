import numpy as np
import os
import pathlib

from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.utils.flow import run_solver, return_c0, return_q, return_rx, make_directory, parse_hyperparameters
from ising.utils.HDF5Logger import HDF5Logger
TOP = pathlib.Path(os.getenv("TOP"))


def test_MIMO(SNR_list, solvers, args):
    nb_runs = int(args.nb_runs)
    Nr = int(args.Nr)
    Nt = int(args.Nt)
    M = int(args.M)
    num_iter = int(args.num_iter)

    hyperparameters = parse_hyperparameters(args, num_iter)
    if hyperparameters["q"] == 0.0:
        change_q = True
        hyperparameters["r_q"] = 1.0
    else:
        hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
        change_q = False
    if hyperparameters["c0"] == 0.0:
        change_c = True
    else:
        change_c = False

    logtop = TOP / "ising/flow/MIMO/logs"
    make_directory(logtop)

    for SNR in SNR_list:
        H, symbols = MU_MIMO(Nt, Nr, M, hyperparameters["seed"])

        for run in range(nb_runs):
            solution_file = logtop  / f"actual_solution_SNR{SNR}_run{run}.log"
            x = np.random.choice(symbols, (Nt,)) + 1j*np.random.choice(symbols, (Nt,))
            model, xtilde = MIMO_to_Ising(H, x, SNR, Nr, Nt, M, hyperparameters["seed"])
            with HDF5Logger(solution_file, schema={"x":np.float16}) as log:
                log.write_metadata(SNR=SNR, run=run, x=xtilde)

            print("Correct solution: ", x)
            s_init = np.random.choice([-1, 1], (model.num_variables,))

            if change_c:
                hyperparameters["c0"] = return_c0(model=model)
            if change_q:
                hyperparameters["q"] = return_q(model)
            for solver in solvers:
                print(f"Run {run} for {solver} with SNR {SNR}")
                logfile = logtop / f"{solver}_SNR{SNR}_run{run}.log"
                run_solver(
                    solver,
                    num_iter=num_iter,
                    s_init=s_init,
                    logfile=logfile,
                    model=model,
                    **hyperparameters
                )
