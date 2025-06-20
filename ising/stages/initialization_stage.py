from ising.stages import LOGGER
from typing import Any
import numpy as np
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

class InitializationStage(Stage):
    """! Stage to initialize the Ising spins and models."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 trail_id: int,
                 config: Any,
                 ising_model: IsingModel,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.trail_id = trail_id
        self.config = config
        self.ising_model = ising_model

    def run(self) -> Any:
        """! Initialize the Ising spins and models."""

        LOGGER.info(f"Initialization stage for trail {self.trail_id}.")
        self.initial_state = np.random.uniform(-1, 1, (self.ising_model.num_variables,))
        self.ising_model = self.ising_model.copy()  # Placeholder for any model-specific initialization
        yield self.initial_state, self.ising_model
