from ising.stages import LOGGER
from typing import Any
import numpy as np
import copy
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel


class NpmosStage(Stage):
    """! Stage to inject the nmos/pmos imbalance on the ising model."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        config: Any,
        ising_model: IsingModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model = ising_model

    def run(self) -> Any:
        """! Inject the imbalance on the J/h matrix of the Ising model."""
        offset_ratio: float = self.config.offset_ratio
        if self.config.offset_type == "negative":
            offset_J = self.offset_negative_on_matrix(self.ising_model.J, offset_ratio)
            offset_h = self.offset_negative_on_matrix(self.ising_model.h, offset_ratio)
        elif self.config.offset_type == "positive":
            offset_J = self.offset_positive_on_matrix(self.ising_model.J, offset_ratio)
            offset_h = self.offset_positive_on_matrix(self.ising_model.h, offset_ratio)
        else:
            LOGGER.debug(f"Unknown offset type: {self.config.offset_type}. Using original J/h matrices.")
            offset_J = self.ising_model.J
            offset_h = self.ising_model.h

        offset_model = IsingModel(
            J=offset_J,
            h=offset_h,
            c=self.ising_model.c,
        )

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = offset_model
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.offset_model = offset_model
            for solver in self.config.solvers:
                for energy_id in range(len(ans.energies[solver])):
                    ans.energies[solver][energy_id] = self.ising_model.evaluate(
                        ans.states[solver][energy_id].astype(np.float32)
                    )
            yield ans, debug_info

    def offset_negative_on_matrix(self, J: np.ndarray, offset_ratio: float) -> np.ndarray:
        """Add uniform offset to the J matrix."""
        offset_J = copy.deepcopy(J)
        offset_J[offset_J < 0] *= 1 + offset_ratio
        return offset_J

    def offset_positive_on_matrix(self, J: np.ndarray, offset_ratio: float) -> np.ndarray:
        """Add uniform offset to the J matrix."""
        offset_J = copy.deepcopy(J)
        offset_J[offset_J > 0] *= 1 + offset_ratio
        return offset_J
