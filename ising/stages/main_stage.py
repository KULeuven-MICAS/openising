from typing import Any
from ising.stages.stage import StageCallable

class MainStage:
    """! Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    def run(self):
        for cme, debug_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
            return cme, debug_info
