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

        LOGGER.debug(f"Initialization stage for trail {self.trail_id}.")
        initialization_seed = self.config.initialization_seed
        if initialization_seed is not None and not self.config.eigenvalue_start:
            np.random.seed(initialization_seed + self.trail_id)
            LOGGER.debug(f"Setting random seed to {initialization_seed + self.trail_id}.")
            self.initial_state = np.random.uniform(-1, 1, (self.ising_model.num_variables,))
        elif self.config.eigenvalue_start:
            coupling = self.ising_model.to_qubo()[0]
            _, eigvecs = np.linalg.eigh(coupling)
            self.initial_state = np.sign(eigvecs[:self.ising_model.num_variables, -2])

        self.ising_model = self.ising_model.copy()  # Placeholder for any model-specific initialization
        if getattr(self.config, "combine_nodes", False):  # combine nodes is deprecated
            nodes_scaling = self.config.nodes_scaling
            orig_num_variables = int(self.ising_model.num_variables / nodes_scaling)
            self.initial_state = np.repeat(np.random.uniform(-1,1, (orig_num_variables,)), nodes_scaling)
        return self.initial_state, self.ising_model
