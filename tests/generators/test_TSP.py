import numpy as np
import pathlib
import os

import ising.generators as gen
from ising.benchmarks.parsers import ATSP_parser

benchmark = pathlib.Path(os.getenv('TOP')) / 'ising/benchmarks/ATSP/ATSP_dummy.txt'
G = ATSP_parser(benchmark)
N = 4
assert len(G.nodes) == N
assert len(G.edges) == 10
A =8.
B=4.
C=2.
model = gen.TSP(G, A=A, B=B, C=C)

J = model.J
h = model.h
BC_mat = np.array([[0., C/4, C/4, C/4],
                   [0, 0., C/4, C/4],
                   [0, 0, 0., C/4],
                   [0, 0, 0, 0.]])
AB_mat = np.array([[B/4, A/8, 0, 0],
                   [0, B/4, A/8, 0],
                   [0, 0, B/4, A/8],
                   [A/8, 0, 0, B/4]])
Jcorrect = -np.block([[BC_mat, AB_mat, AB_mat.T, AB_mat.T],
                     [np.zeros((4,4)), BC_mat, AB_mat, AB_mat],
                     [np.zeros((4, 4)), np.zeros((4, 4)), BC_mat, AB_mat],
                     [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)), BC_mat]])
hBC = (N-2)/2*(B+C)
hA = A/2
hcorrect = -np.array([hBC + hA, hBC + hA, hBC + hA, hBC + hA, hBC + 2*hA, hBC + 2*hA, hBC + 2*hA, hBC + 2*hA, \
                      hBC + 2*hA, hBC + 2*hA, hBC + 2*hA, hBC + 2*hA, hBC + hA, hBC + hA, hBC + hA, hBC + hA])
assert np.shape(h) == (N*N,)
assert np.shape(J) == (N*N, N*N)
assert not np.all(J - Jcorrect)
assert not np.all(h - hcorrect)
