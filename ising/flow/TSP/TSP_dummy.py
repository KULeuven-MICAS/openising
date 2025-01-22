import argparse
import os
import pathlib

from ising.utils.threading import make_solvers_thread, make_Gurobi_thread

TOP = pathlib.Path(os.getenv("TOP"))

parser=  argparse.ArgumentParser()


