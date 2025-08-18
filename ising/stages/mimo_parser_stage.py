import numpy as np
import time

from ising.stages import LOGGER, TOP
from typing import Any
from ising.stages.stage import Stage, StageCallable
from ising.generators.MIMO import MIMO_to_Ising
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

        nb_trials = int(self.config.nb_trials)
        LOGGER.debug(f"Parsing MIMO benchmark: {self.benchmark_filename}")
        Nt, Nr = int(self.config.Nt), int(self.config.Nr)
        M = int(self.config.M)
        H, symbols = self.MU_MIMO(Nt=Nt, Nr=Nr,
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


    def MU_MIMO(self, Nt: int, Nr: int, M: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Generates a MU-MIMO model using section IV-A of [this paper](https://arxiv.org/pdf/2002.02750).
        This is consecutively transformed into an Ising model.

        Args:
            Nt (int): The amount of users.
            Nr (int): The amount of antennas at the Base Station.
            M (int): the considered QAM scheme.
            seed (int, optional): The seed for the random number generator. Defaults to 1.

        Returns:
            H,symbols (tuple[np.ndarray, np.ndarray]): The generated transfer function matrix and the symbols.
        """
        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        if M==2:
            # BPSK scheme
            symbols = np.array([-1, 1])
            r = 1
        else:
            r = int(np.ceil(np.log2(np.sqrt(M))))
            symbols = np.concatenate(
                ([-np.sqrt(M) + i for i in range(1, 2 + 2 * r, 2)], [np.sqrt(M) - i for i in range(1, 2 + 2 * r, 2)])
            )

        phi_u     = 120 * (np.random.random((10, Nt)) - 0.5)
        phi_u.sort()
        mean_phi  = np.mean(phi_u, axis=0)
        sigma_phi = np.random.normal(0, 1, (Nt,))

        # H = np.random.random((Nr, Nt)) + 1j*np.random.random((Nr, Nt))
        H = np.zeros((Nr, Nt), dtype='complex_')
        for i in range(Nt):
            C     = np.zeros((Nr, Nr), dtype="complex_")
            phi   = mean_phi[i]
            sigma = sigma_phi[i]
            for m in range(Nr):
                for n in range(Nr):
                    d = self.spacing_BS_antennas(m, n)
                    C[m, n] = np.exp(2*np.pi*1j*d*np.sin(phi))* np.exp(
                        -(sigma**2) / 2 * (2 * np.pi * d * np.cos(phi)) ** 2
                    )
            D, V = np.linalg.eig(C)
            hu = V @ np.diag(D)**0.5 @ V.conj().T @ (np.random.normal(0, 1, (Nr,)) + 1j*np.random.normal(0, 1, (Nr,)))
            H[:, i] = hu

        return H, symbols

    def spacing_BS_antennas(self,m, n):
        return np.abs(m - n)

