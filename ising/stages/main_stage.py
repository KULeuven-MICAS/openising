from typing import Any
from ising.stages.stage import StageCallable
from ising.stages.simulation_stage import Ans

class MainStage:
    """! Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    def run(self)-> tuple[Ans, Ans]:
        """Main stage to set up the process of all other stages

        @rtype: tuple[Ans, Ans]
        @return: A tuple containing the answer and debug info from the simulation.
        """
        for cme, debug_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
            return cme, debug_info
