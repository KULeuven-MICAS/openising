import numpy as np
from solver import Solver

class SCA(Solver):
    def __init__(self, sigma:np.ndarray, model, file, S, T, r_t, q, r_q, verbose):
            Solver.__init__(sigma, model, file, verbose)
            self.S = S
            self.T = T
            self.r_t = r_t
            self.q = q
            self.r_q = r_q
    def run():
          pass
