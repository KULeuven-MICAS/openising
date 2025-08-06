from ising.stages import LOGGER
from typing import Any
import numpy as np
import copy
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

class NoiseStage(Stage):
    """! Stage to inject the noise on the ising model."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 ising_model: IsingModel | None = None,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model = ising_model

    def run(self) -> Any:
        """! Inject the noise on the J/h matrix of the Ising model."""
        if self.config.device_noise:
            noise_level: float = self.config.noise_level
            assert 0 <= noise_level <= 1, f"noise level {noise_level} is not in [0, 1]."
            original_J = copy.deepcopy(self.ising_model.J)
            noisy_J = self.noise_on_matrix(original_J, noise_level)

            original_h = copy.deepcopy(self.ising_model.h)
            noisy_h = self.noise_on_matrix(original_h, noise_level)
            noisy_model = IsingModel(
                J=noisy_J,
                h=noisy_h,
                c=self.ising_model.c,
            )
        else:
            LOGGER.debug("Noise is disabled, using original J/h matrices.")
            noisy_model = self.ising_model

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = noisy_model
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.noisy_model = noisy_model
            for energy_id in range(len(ans.energies)):
                ans.energies[energy_id] = self.ising_model.evaluate(ans.states[energy_id])
            yield ans, debug_info

    def noise_on_matrix(self, J: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to the J matrix."""
        noise = np.random.normal(loc=1, scale=noise_level, size=J.shape)
        noise[noise < 0] == 0  # Sign flipping is assumed to be impossible
        assert np.allclose(J, np.triu(J)), "J must be an upper-triangular matrix."
        noisy_J = J * noise
        return noisy_J
