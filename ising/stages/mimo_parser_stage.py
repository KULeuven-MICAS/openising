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

        dummy_creator = self.config.dummy_creator if hasattr(self.config, "dummy_creator") else False
        if dummy_creator:
            dummy_dict = self.kwargs.get("dummy_dict", {})
            H = dummy_dict.get("H", None)
            x = dummy_dict.get("x_collect", None)
            M = dummy_dict.get("M", None)
            ant_num = dummy_dict.get("ant_num", None)
            user_num = dummy_dict.get("user_num", None)
            snr = dummy_dict.get("SNR", None)
            mimo_seed = dummy_dict.get("seed", None)
        else:
            H, x, M, ant_num, user_num = self.parse_MIMO(self.benchmark_filename)
            snr = int(self.config.SNR)
            mimo_seed = int(self.config.mimo_seed)

        if hasattr(self.config, "nb_trials"):
            case_num = self.config.nb_trials
            case_num = min(case_num, x.shape[1])
        else:
            case_num = x.shape[1]
        self.kwargs["config"] = self.config
        self.kwargs["best_found"] = 0.0

        ans_all = Ans()
        ans_all.MIMO = []
        diff = np.zeros((2*user_num, case_num))
        for run in range(case_num):
            xi = x[:, run]
            ising_model, x_tilde, _ = self.MIMO_to_Ising(
                H, xi, snr, user_num, ant_num, M, mimo_seed)
            self.kwargs["ising_model"] = ising_model
            self.kwargs["x_tilde"] = x_tilde
            self.kwargs["M"] = M
            self.kwargs["run_id"] = run
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

            ans: Ans
            debug_info: Ans
            ans, debug_info = next(sub_stage.run())
            ans_all.MIMO.append(ans)
            diff[:, run] = ans.difference

        # calc ber per trail
        ans_all.ber_of_trails = np.sum(np.abs(diff) / 2, axis=0) / (np.log2(M)*user_num)

        # calc ber per user
        array_mid = diff.shape[0] // 2
        diff_real_half = diff[0:array_mid, :]
        diff_imag_half = diff[array_mid:, :]
        diff_of_users = np.hstack((diff_real_half, diff_imag_half))
        ans_all.ber_of_users = np.sum(np.abs(diff_of_users) / 2, axis=1) / (np.log2(M)*case_num)

        ans_all.BER = np.mean(ans_all.ber_of_trails) # same as np.mean(ans_all.ber_of_users)
        LOGGER.info("BER/case: %s, BER/user: %s, mean: %s", ans_all.ber_of_trails, ans_all.ber_of_users, ans_all.BER)

        yield ans_all, debug_info

    @staticmethod
    def parse_MIMO(benchmark:pathlib.Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """! Parses the MIMO benchmark from the given file.

        @param benchmark (pathlib.Path): the path to the benchmark to parse.

        @return H (np.ndarray): the transfer function matrix of the MIMO system.
        @return x (np.ndarray): all the input signals that where sent.
        @return M (int): the considered QAM scheme.
        @return ant_num (int): amount of user antennas for the MIMO problem.
        @return user_num (int): amount of receiver antennas for the MIMO problem.
        """

        with benchmark.open() as f:
            lines = f.readlines()

        # Parse dimensions
        M: int = int(lines[0].split()[1]) # modulation scheme QAM-M
        ant_num: int = int(lines[1].split()[1])
        user_num: int = int(lines[2].split()[1])

        # Locate sections
        h_real_idx = lines.index("REAL\n", lines.index("H\n")) + 1
        h_imag_idx = lines.index("IMAG\n", lines.index("H\n")) + 1
        sig_real_idx = lines.index("REAL\n", lines.index("SIGNALS\n")) + 1
        sig_imag_idx = lines.index("IMAG\n", lines.index("SIGNALS\n")) + 1

        # Read H (matrix ant_num x user_num)
        H_real = []
        for i in range(ant_num):
            H_real.append([float(x) for x in lines[h_real_idx + i].split()])
        H_imag = []
        for i in range(ant_num):
            H_imag.append([float(x) for x in lines[h_imag_idx + i].split()])
        H = np.array(H_real) + 1j * np.array(H_imag)

        # Read SIGNALS (flattened array, length = M * something)
        signals_real = []
        for i in range(user_num):
            signals_real.append([float(x) for x in lines[sig_real_idx + i].split()])
        signals_imag = []
        for i in range(user_num):
            signals_imag.append([float(x) for x in lines[sig_imag_idx + i].split()])
        x = np.array(signals_real) + 1j * np.array(signals_imag)

        return H, x, M, ant_num, user_num

    @staticmethod
    def MIMO_to_Ising(
        H: np.ndarray, x: np.ndarray, SNR: float, user_num: int, ant_num: int, M: int, seed:int=0
    ) -> tuple[IsingModel, np.ndarray, np.ndarray]:
        """!Transforms the MIMO model into an Ising model.

        @param H (np.ndarray): The transfer function matrix.
        @param x (np.ndarray): the input signal.
        @param T (np.ndarray): the transformation matrix to transform the input signal to Ising format.
        @param SNR (float): the signal to noise ratio.
        @param user_num (int): the amount of input signals.
        @param ant_num (int): the amount of output signals.
        @param M (int): the considered QAM scheme.
        @param seed (int, optional): The seed for the random noise generation. Defaults to 0.

        @return model (IsingModel): the generated Ising model.
        @return xtilde (np.ndarray): the real version of the input symbols.
        @return ytilde (np.ndarray): the real version of the output symbols.
        """
        is_bpsk = np.linalg.norm(np.imag(x)) == 0

        if is_bpsk:
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
        var_noise = np.sqrt(np.mean(power_x) / SNR)
        n = var_noise*(np.random.randn(ant_num) + 1j * np.random.randn(ant_num)) / (np.sqrt(2)) # noise

        # Compute the received symbols
        y = H @ x + n
        ytilde = np.block([np.real(y), np.imag(y)])

        Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])

        T = np.block([2**(r-i)*np.eye(Ny, Nx) for i in range(1, r+1)])

        if is_bpsk:
            xtilde = x
        else:
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

