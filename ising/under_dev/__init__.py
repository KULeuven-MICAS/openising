from argparse import Namespace
import numpy as np
import os

from ising.stages.main_stage import MainStage
from ising.stages.simulation_stage import SimulationStage
from ising.stages.model.ising import IsingModel
from ising.stages.maxcut_parser_stage import MaxcutParserStage
from ising.stages.tsp_parser_stage import TSPParserStage
from ising.stages.atsp_parser_stage import ATSPParserStage
from ising.stages.qkp_parser_stage import QKPParserStage
from ising.stages.mimo_parser_stage import MIMOParserStage

nb_cores = 16
os.sched_setaffinity(0, range(nb_cores))
os.environ["MKL_NUM_THREADS"] = str(nb_cores//3)
os.environ["NUMEXPR_NUM_THREADS"] = str(nb_cores//3)
os.environ["OMP_NUM_THREADS"] = str(nb_cores//3)
os.environ["OPENBLAS_NUM_THREADS"] = str(nb_cores//3)

sim_stage = SimulationStage([MainStage], config=Namespace(benchmark="ising/G16.txt"), ising_model=IsingModel(np.zeros((2,2)), np.zeros((2,))))
MaxCutParser = MaxcutParserStage([MainStage], config=Namespace(benchmark=""))
TSPParser = TSPParserStage([MainStage], config=Namespace(benchmark=""))
ATSPParser = ATSPParserStage([MainStage], config=Namespace(benchmark=""))
QKPParser = QKPParserStage([MainStage], config=Namespace(benchmark=""))
MIMOParser = MIMOParserStage([MainStage], config=Namespace(benchmark="ising/benchmarks/MIMO/benchmark_4x4.txt"))
