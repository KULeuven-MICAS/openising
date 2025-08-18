import numpy as np

from typing import Any
from ising.stages.stage import Stage, StageCallable

class MIMOBerCalcStage(Stage):
    """! Stage to calculate the BER for MIMO benchmark workload."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 x_tilde:np.ndarray,
                 M: int,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.x_tilde = x_tilde
        self.M = M

    def run(self) -> Any:
        """! Calculate BER for all the different trials."""
        self.kwargs["config"] = self.config
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        r = int(np.ceil(np.log2(np.sqrt(self.M))))

        N = np.shape(self.x_tilde)[0]

        # Compute the calculated symbols
        T = np.block([[2 ** (r - i) * np.eye(N) for i in range(1, r + 1)]])
        min_en = np.inf
        best_found = 0
        for ans, debug_info in sub_stage.run():
            for i in range(len(ans.states)):
                energy = ans.energies[i]
                if energy < min_en:
                    min_en = energy
                    best_found = i

            # Compute the BER
            state = ans.states[best_found]
            x_optim = T @ (state + np.ones((r * N,))) - (np.sqrt(self.M) - 1) * np.ones((N,))
            ans.difference = self.x_tilde - x_optim
            ans.lowest_energy = min_en
            ans.lowest_energy_state = ans.states[best_found]

            return ans, debug_info
