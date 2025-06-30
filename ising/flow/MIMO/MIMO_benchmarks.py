import numpy as np
import pathlib

from ising.flow import TOP, LOGGER
from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.utils.flow import parse_hyperparameters, run_solver
from ising.utils.helper_functions import return_q, return_rx, return_c0, make_directory
from ising.utils.HDF5Logger import HDF5Logger, return_metadata
from ising.generators.MIMO import compute_difference
from ising.solvers.Gurobi import Gurobi


def test_MIMO(SNR_list, solvers, args):
    nb_runs = int(args.nb_runs)
    Nr = int(args.Nr)
    Nt = int(args.Nt)
    M = int(args.M)
    num_iter = int(args.num_iter)

    use_gurobi = bool(int(args.use_gurobi))
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
    LOGGER.debug(f"Logtop: {logtop}")
    make_directory(logtop)
    H, symbols = MU_MIMO(Nt, Nr, M, hyperparameters["seed"])
    for SNR in SNR_list:
        LOGGER.info(f"running for SNR {SNR}")
        for run in range(nb_runs):
            x = np.random.choice(symbols, (Nt,)) + 1j*np.random.choice(symbols, (Nt,))
            model, xtilde = MIMO_to_Ising(H, x, SNR, Nr, Nt, M, hyperparameters["seed"])

            if use_gurobi:
                gurobi_file = logtop / f"Gurobi_SNR{SNR}_run{run}.log"
                Gurobi().solve(model, gurobi_file)
                add_bit_error_rate([gurobi_file], xtilde, M, SNR)

            if change_c:
                hyperparameters["c0"] = return_c0(model=model)
            if change_q:
                hyperparameters["q"] = return_q(model)
            current_logfiles = [logtop / f"{solver}_SNR{SNR}_run{run}.log" for solver in solvers]

            for solver in solvers:
                s_init = np.random.choice([-1, 1], (model.num_variables,))
                logfile = logtop / f"{solver}_SNR{SNR}_run{run}.log"
                run_solver(
                    solver,
                    num_iter=num_iter,
                    s_init=s_init,
                    logfile=logfile,
                    model=model,
                    **hyperparameters
                )
                current_logfiles.append(logfile)

            add_bit_error_rate(current_logfiles, xtilde, M, SNR)


def add_bit_error_rate(logfiles:list[pathlib.Path], xtilde:np.ndarray, M:int, SNR:int) -> None:
    """Adds the bit error rate to the logfiles.

    Args:
        logfiles (list[pathlib.Path]): list of all the logfiles that solve the problem with solution xtilde.
        xtilde (np.ndarray): the solution to the MU-MIMO problem.
        M (int): the modulation order.
    """
    for logfile in logfiles:
        sigma_optim = return_metadata(logfile, "solution_state")
        BER = compute_difference(sigma_optim, xtilde, M)
        with HDF5Logger(logfile,schema=dict(), mode="a") as logger:
            logger.write_metadata(BER=BER, SNR=SNR)


