import numpy as np

from typing import Any
from ising.stages.stage import Stage, StageCallable


def gray_bit_errors(x_tx: np.ndarray, x_rx: np.ndarray, M: int) -> np.ndarray:
    """! Exact number of bit errors per real dimension, assuming Gray mapping.

    Square M-QAM is Gray-mapped independently on the I and Q axes, so each real dimension carries a
    sqrt(M)-PAM symbol with log2(sqrt(M)) Gray-coded bits (BPSK: 2-PAM, 1 bit).

    @param x_tx: transmitted real symbol levels (any shape), values in {-(L-1), ..., -1, +1, ..., L-1}.
    @param x_rx: decided real symbol levels, same shape as x_tx.
    @param M: the QAM order (2 for BPSK).
    @return: integer array of the same shape with the number of differing bits per real symbol.
    """
    L = 2 if M == 2 else int(round(np.sqrt(M)))
    k_tx = np.rint((np.real(x_tx) + L - 1) / 2).astype(np.int64)
    k_rx = np.rint((np.real(x_rx) + L - 1) / 2).astype(np.int64)
    assert np.all((k_tx >= 0) & (k_tx < L)), "Transmitted symbols are not on the constellation"
    # Decisions outside the constellation are mapped to the nearest outer level
    k_rx = np.clip(k_rx, 0, L - 1)
    xor = (k_tx ^ (k_tx >> 1)) ^ (k_rx ^ (k_rx >> 1))  # XOR of the Gray codes
    # popcount (L <= 256 for M <= 65536, so the codes fit in uint8)
    return np.unpackbits(xor.astype(np.uint8)[..., np.newaxis], axis=-1).sum(axis=-1)


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
        if hasattr(config, "is_hamming_encoding"):
            self.is_hamming_encoding = config.is_hamming_encoding
        else:
            self.is_hamming_encoding = False

    def run(self) -> Any:
        """! Calculate BER for all the different trials."""
        self.kwargs["config"] = self.config
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        if self.M == 2:
            r = 1
        else:
            if self.is_hamming_encoding: # with hamming encoding
                r = int(np.sqrt(self.M) - 1)
            else: # with binary encoding
                r = int(np.ceil(np.log2(np.sqrt(self.M))))

        N = np.shape(self.x_tilde)[0]

        # Compute the calculated symbols
        if self.is_hamming_encoding: # with hamming encoding
            T = np.block([[np.eye(N) for _ in range(r)]])
        else: # with binary encoding
            T = np.block([[2 ** (r - i) * np.eye(N) for i in range(1, r + 1)]])
        for ans, debug_info in sub_stage.run():
            ans.difference = {solver: None for solver in self.config.solvers}
            ans.bit_errors = {solver: None for solver in self.config.solvers}
            ans.lowest_energy = {solver: None for solver in self.config.solvers}
            ans.lowest_energy_state = {solver: None for solver in self.config.solvers}
            for solver in self.config.solvers:
                min_en = np.inf
                best_found = 0
                for i in range(len(ans.states[solver])):
                    energy = ans.energies[solver][i]
                    if energy < min_en:
                        min_en = energy
                        best_found = i

                # Compute the BER
                state = ans.states[solver][best_found]
                if self.M == 2:
                    # BPSK scheme
                    x_optim = T @ (state + np.ones((r * N,))) - np.ones((N,))
                else:
                    # QAM scheme
                    x_optim = T @ (state + np.ones((r * N,))) - (np.sqrt(self.M) - 1) * np.ones((N,))
                ans.difference[solver] = self.x_tilde - x_optim
                ans.bit_errors[solver] = gray_bit_errors(self.x_tilde, x_optim, self.M)
                ans.lowest_energy[solver] = min_en
                ans.lowest_energy_state[solver] = ans.states[solver][best_found]

            yield ans, debug_info
