import logging
from typing import Any
import sys
from ising.stages.stage import StageCallable
from ising.stages.main_stage import MainStage
from ising.stages.dummy_creator_stage import DummyCreatorStage
from ising.stages.config_parser_stage import ConfigParserStage
from ising.stages.maxcut_parser_stage import MaxcutParserStage
from ising.stages.tsp_parser_stage import TSPParserStage
from ising.stages.atsp_parser_stage import ATSPParserStage
from ising.stages.mimo_parser_stage import MIMOParserStage
from ising.stages.mimo_ber_calc_stage import MIMOBerCalcStage
from ising.stages.tsp_energy_calc_stage import TSPEnergyCalcStage
from ising.stages.simulation_stage import SimulationStage
from ising.stages.initialization_stage import InitializationStage
from ising.stages.quantization_stage import QuantizationStage
from ising.stages.npmos_stage import NpmosStage

def get_hamiltonian_energy(
    create_dummy_problem: bool = False,
    problem_type: str = "TSP",
    config_path: str = "./ising/inputs/config/config_tsp.yaml",
    logging_level: int = logging.INFO,
) -> tuple[float, str, Any]:
    """! API: simulate and evaluate the Hamiltonian of an Ising model.
    @param config_path: Path to the configuration file.
    """

    # Initialize the logger
    logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

    # Decide on the parser stage
    if create_dummy_problem:
        parser_stage = DummyCreatorStage
        logging.info("Using DummyCreatorStage to create a dummy problem.")
    elif problem_type == "Maxcut":
        parser_stage = MaxcutParserStage
    elif problem_type == "TSP":
        parser_stage = TSPParserStage
    elif problem_type == "ATSP":
        parser_stage = ATSPParserStage
    elif problem_type == "MIMO":
        parser_stage = MIMOParserStage
    else:
        logging.error(f"Parser for {problem_type} is not implemented.")
        raise NotImplementedError(f"Parser for {problem_type} is not implemented.")

    # Decide on the energy calculation stage
    if problem_type == "TSP" or problem_type == "ATSP":
        energy_calc_stage = TSPEnergyCalcStage
    elif problem_type == "MIMO":
        energy_calc_stage = MIMOBerCalcStage
    else:
        energy_calc_stage = None

    # Add simulation stages
    stages = [
        ConfigParserStage,  # Parses the configuration file
        DummyCreatorStage,  # Creates a dummy Ising model if needed
        parser_stage,  # Parses the specific problem into an Ising graph model
        energy_calc_stage,  # Calculates the energy for the problems
        NpmosStage,  # Injects NMOS/PMOS imbalance if needed
        QuantizationStage,  # Quantizes the Ising model if needed
        SimulationStage,  # Runs the simulation on the Ising model
        InitializationStage,  # Initializes the Ising spins and model
    ]

    stage_callables: list[StageCallable] = [s for s in stages if s is not None]

    # Initialize the MainStage as entry point
    mainstage = MainStage(
        list_of_callables=stage_callables,
        config_path=config_path,
        problem_type=problem_type,
    )

    # Launch the MainStage
    ans, debug_info = mainstage.run()

    return ans, debug_info
