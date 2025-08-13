import numpy as np

from ising.stages import LOGGER, TOP
from typing import Any
from ising.stages.stage import Stage, StageCallable
from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.stages.simulation_stage import Ans

class MIMOParserStage(Stage):
    """! Stage to parse the MIMO benchmark workload."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / "MIMO.mimo"

    def run(self) -> Any:
        """! Parse the MIMO benchmark workload."""
        self.initialization_seed = self.config.initialization_seed
        if self.initialization_seed is not None:
            np.random.seed(self.initialization_seed)

        nb_trials = int(self.config.nb_runs)
        self.config.nb_runs = 1
        LOGGER.debug(f"Parsing MIMO benchmark: {self.benchmark_filename}")
        Nt, Nr = int(self.config.Nt), int(self.config.Nr)
        M = int(self.config.M)
        H, symbols = MU_MIMO(Nt=Nt, Nr=Nr,
                             M=M, seed=int(self.config.seed))

        self.kwargs["config"] = self.config
        self.kwargs["best_found"] = 0.0

        x = np.random.choice(symbols, size=(int(self.config.Nt), nb_trials)) + \
            1j * np.random.choice(symbols, size=(int(self.config.Nt), nb_trials))
        self.x_tilde = np.zeros((2*Nt, nb_trials))
        ans_all = Ans()
        ans_all.MIMO = []
        diff = np.zeros((2*Nt, nb_trials))
        for run in range(nb_trials):
            xi = x[:, run]
            ising_model, x_tilde, _ = MIMO_to_Ising(H, xi, int(self.config.SNR), Nr, Nt, M, int(self.config.seed))
            self.x_tilde[:, run] = x_tilde
            self.kwargs["ising_model"] = ising_model
            self.kwargs["x_tilde"] = x_tilde
            self.kwargs["M"] = M
            self.kwargs["run_id"] = run
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

            ans, debug = sub_stage.run()
            ans_all.MIMO.append(ans)
            diff[:, run] = ans.difference

        ans_all.BER = np.sum(np.abs(diff) / 2, axis=0) / (np.sqrt(M)*Nt)
        LOGGER.info("BER: %s", ans_all.BER)
        ans_all.BER = np.mean(ans_all.BER)

        yield ans_all, debug
        # sub_stage = self.list_of_callables[1](self.list_of_callables[2:], **self.kwargs)

