import pathlib
import numpy as np

from ising.utils.HDF5Logger import HDF5Logger, return_metadata
from ising.generators.MIMO import compute_difference

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
        with HDF5Logger(logfile, schema={"x":float}, mode="a") as logger:
            logger.write_metadata(BER=BER, SNR=SNR)
