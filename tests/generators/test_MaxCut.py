import os
import pathlib
import numpy as np

import ising.generators as gen
from ising.benchmarks.parsers import G_parser


TOP = os.getenv('TOP')
benchmark = pathlib.Path(TOP) / 'ising/benchmarks/G/G_dummy.txt'
G = G_parser(benchmark)
N = 5
assert len(G.nodes) == N
assert len(G.edges) == 6

model = gen.MaxCut(G)

J = model.J
h = model.h
Jcorrect = np.array([[0, -1/2, -1/2, 0, 0],
                     [0, 0, 0, -1/2, 0],
                     [0, 0, 0, -1/2, -1/2],
                     [0, 0, 0, 0, -1/2],
                     [0, 0, 0, 0, 0]])
assert np.shape(h) == (N,)
assert np.linalg.norm(h) == 0.
assert np.shape(J) == (N, N)
assert not np.all(J - Jcorrect)
