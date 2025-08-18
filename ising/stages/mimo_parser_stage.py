import numpy as np
import time
import pathlib

from ising.stages import LOGGER, TOP
from typing import Any
from ising.stages.stage import Stage, StageCallable
from ising.stages.simulation_stage import Ans
from ising.stages.model.ising import IsingModel

class MIMOParserStage(Stage):
    """! Stage to parse the MIMO benchmark workload."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 config: Any,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / config.benchmark

    def run(self) -> Any:
        """! Parse the MIMO benchmark workload."""
        LOGGER.debug(f"Parsing MIMO benchmark: {self.benchmark_filename}")

        Nt, Nr = int(self.config.Nt), int(self.config.Nr)
        M = int(self.config.M)
        H, x, M = self.parse_MIMO(self.benchmark_filename)
        nb_trials = x.shape[1]
        self.kwargs["config"] = self.config
        self.kwargs["best_found"] = 0.0

        self.x_tilde = np.zeros((2*Nt, nb_trials))
        ans_all = Ans()
        ans_all.MIMO = []
        diff = np.zeros((2*Nt, nb_trials))
        for run in range(nb_trials):
            xi = x[:, run]
            ising_model, x_tilde, _ = self.MIMO_to_Ising(H, xi, int(self.config.SNR), Nr, Nt, M, int(self.config.seed))
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

    @staticmethod
    def parse_MIMO(benchmark:pathlib.Path) -> tuple[np.ndarray, np.ndarray, int]:
        """! Parses the MIMO benchmark from the given file.

        @param benchmark (pathlib.Path): the path to the benchmark to parse.

        @return H (np.ndarray): the transfer function matrix of the MIMO system.
        @return x (np.ndarray): all the input signals that where sent.
        @return M (int): the considered QAM scheme.
        """
        M = None
        Nt = None
        Nr = None

        H = None
        x = None

        transfer_stage = False
        signals_stage = False

        Real_stage = False
        Imag_stage = False
        with benchmark.open() as f:
            for line in f:
                parts = line.split(" ")
                if transfer_stage:
                    if parts[0] == "REAL":
                        Real_stage = True
                        Imag_stage = False
                        index = 0
                        continue
                    elif parts[0] == "IMAG":
                        Real_stage = False
                        Imag_stage = True
                        index = 0
                        continue

                    if Real_stage:
                        H[index, :] = np.array([float(h) for h in parts])
                        index += 1
                    elif Imag_stage:
                        H[index, :] += 1j * np.array([float(h) for h in parts])
                        index += 1
                elif signals_stage:
                    if parts[0] == "REAL":
                        Real_stage = True
                        Imag_stage = False
                        continue
                    elif parts[0] == "IMAG":
                        Real_stage = False
                        Imag_stage = True
                        index = 0
                        continue

                    if Real_stage:
                        x = np.append(x, np.array([float(symbol) for symbol in parts]), axis=1)
                    elif Imag_stage:
                        x[index, :] += 1j * np.array([float(symbol) for symbol in parts])
                        index += 1
                elif parts[0] == "H":
                    transfer_stage = True
                elif parts[0] == "SIGNALS":
                    signals_stage = True
                    transfer_stage = False
                elif parts[0] == "M":
                    M = int(parts[1])
                elif parts[0] == "Nt":
                    Nt = int(parts[1])
                elif parts[0] == "Nr":
                    Nr = int(parts[1])
                    H = np.zeros((Nr, Nt), dtype='complex_')
                    x = np.zeros((Nt, 0), dtype='complex_')
        return H, x, M

    @staticmethod
    def MIMO_to_Ising(
        H: np.ndarray, x: np.ndarray, SNR: float, Nr: int, Nt: int, M: int, seed:int=0
    ) -> tuple[IsingModel, np.ndarray]:
        """!Transforms the MIMO model into an Ising model.

        @param H (np.ndarray): The transfer function matrix.
        @param x (np.ndarray): the input signal.
        @param T (np.ndarray): the transformation matrix to transform the input signal to Ising format.
        @param SNR (float): the signal to noise ratio.
        @param Nr (int): the amount of input signals.
        @param Nt (int): the amount of output signals.
        @param M (int): the considered QAM scheme.
        @param seed (int, optional): The seed for the random number generator. Defaults to 0.

        @return model (IsingModel): the generated Ising model.
        @return xtilde (np.ndarray): the real version of the input symbols.
        @return ytilde (np.ndarray): the real version of the output symbols.
        """
        if np.linalg.norm(np.imag(x)) == 0:
            # BPSK scheme
            r = 1
            Nx = np.shape(x)[0]
            Ny = 2*Nx
        else:
            r = int(np.ceil(np.log2(np.sqrt(M))))
            Nx = np.shape(x)[0]*2
            Ny = Nx

        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Compute the amplitude of the noise
        power_x = (np.abs(x)**2)
        SNR = 10 ** (SNR / 10)
        var_noise = np.sqrt(power_x / SNR)
        n = var_noise*(np.random.randn(Nr) + 1j * np.random.randn(Nr)) / (np.sqrt(2))

        # Compute the received symbols
        y = H @ x + n
        ytilde = np.block([np.real(y), np.imag(y)])

        Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])

        T = np.block([2**(r-i)*np.eye(Ny, Nx) for i in range(1, r+1)])

        xtilde = np.block([np.real(x), np.imag(x)])

        ones_end = np.eye(Ny, Nx) @ np.ones((Nx,))
        constant = ytilde.T@ytilde - 2*ytilde.T @ Htilde @ (T@np.ones((r*Nx,)) - \
                                            (np.sqrt(M)-1)*ones_end)

        bias = 2*(ytilde - Htilde@(T@np.ones((r*Nx,))-(np.sqrt(M)-1)*ones_end))
        bias = bias.T @ Htilde @ T
        coupling = -2*T.T @ Htilde.T @ Htilde @ T
        diagonal = np.diag(coupling)
        constant -= np.sum(diagonal)/2

        coupling = np.triu(coupling, k=1)
        return IsingModel(coupling, bias, constant, name=f"MIMO_{SNR}"), xtilde, ytilde

