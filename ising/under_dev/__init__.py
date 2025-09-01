from argparse import Namespace
import numpy as np

from ising.stages.main_stage import MainStage
from ising.stages.simulation_stage import SimulationStage
from ising.stages.model.ising import IsingModel
from ising.stages.maxcut_parser_stage import MaxcutParserStage
from ising.stages.tsp_parser_stage import TSPParserStage
from ising.stages.atsp_parser_stage import ATSPParserStage


sim_stage = SimulationStage([MainStage], config=Namespace(benchmark="ising/G16.txt"), ising_model=IsingModel(np.zeros((2,2)), np.zeros((2,))))
MaxCutParser = MaxcutParserStage([MainStage], config=Namespace(benchmark=""))
TSPParser = TSPParserStage([MainStage], config=Namespace(benchmark=""))
ATSPParser = ATSPParserStage([MainStage], config=Namespace(benchmark=""))
