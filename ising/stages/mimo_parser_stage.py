import numpy as np
import time
import pathlib

from ising.stages import LOGGER, TOP
from typing import Any
from ising.stages.stage import Stage, StageCallable
from ising.stages.simulation_stage import Ans
from ising.stages.model.ising import IsingModel
from ising.solvers.MIMO.ZF import ZF


class MIMOParserStage(Stage):
    """! Stage to parse the MIMO benchmark workload."""

    def __init__(self, list_of_callables: list[StageCallable], *, config: Any, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_filename = TOP / config.benchmark
        if hasattr(config, "is_hamming_encoding"):
            self.is_hamming_encoding = config.is_hamming_encoding
        else:
            self.is_hamming_encoding = False

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
        ans_all.ber_of_trials = {solver: None for solver in self.config.solvers}
        ans_all.ber_of_users = {solver: None for solver in self.config.solvers}
        ans_all.BER = {solver: None for solver in self.config.solvers}
        ans_all.MIMO = []
        ans_all.computation_time = {solver: [] for solver in self.config.solvers}
        is_bpsk = M == 2
        if is_bpsk:
            diff = {solver: np.zeros((user_num, case_num)) for solver in self.config.solvers}
        else:
            diff = {solver: np.zeros((2 * user_num, case_num)) for solver in self.config.solvers}
        y = np.zeros((ant_num, case_num), dtype=np.complex128)
        for run in range(case_num):
            xi = x[:, run]
            ising_model, x_tilde, yi = self.MIMO_to_Ising(
                H, xi, snr, user_num, ant_num, M, mimo_seed, is_hamming_encoding=self.is_hamming_encoding
            )
            y[:, run] = yi
            self.kwargs["ising_model"] = ising_model
            self.kwargs["x_tilde"] = x_tilde
            self.kwargs["M"] = M
            self.kwargs["run_id"] = run
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

            ans: Ans
            debug_info: Ans
            ans, debug_info = next(sub_stage.run())
            ans_all.MIMO.append(ans)
            for solver in self.config.solvers:
                ans_all.computation_time[solver] += ans.computation_time[solver]
                diff[solver][:, run] = ans.difference[solver]
        if self.config.use_ZF:
            x_hat_ZF = ZF().solve(H, y, M)
            diff_ZF = x[:, :case_num] - x_hat_ZF
            ans_all.ber_of_trials["ZF"] = np.sum(
                np.abs(np.real(diff_ZF)) / 2 + np.abs(np.imag(diff_ZF)) / 2, axis=0
            ) / (np.log2(M) * user_num)
            ans_all.ber_of_users["ZF"] = np.sum((np.abs(np.real(diff_ZF)) + np.abs(np.imag(diff_ZF))) / 2, axis=1) / (
                np.log2(M) * case_num
            )
            ans_all.BER["ZF"] = np.mean(ans_all.ber_of_users["ZF"])

        for solver in self.config.solvers:
            # calc ber per trail
            ans_all.ber_of_trials[solver] = np.sum(np.abs(diff[solver]) / 2, axis=0) / (np.log2(M) * user_num)
            # calc ber per user
            array_mid = diff[solver].shape[0] // 2
            diff_real_half = diff[solver][0:array_mid, :]
            diff_imag_half = diff[solver][array_mid:, :]
            diff_of_users = np.hstack((diff_real_half, diff_imag_half))
            ans_all.ber_of_users[solver] = np.sum(np.abs(diff_of_users) / 2, axis=1) / (np.log2(M) * case_num)
            ans_all.BER[solver] = np.mean(ans_all.ber_of_users[solver])
        ans_all.operation_count = ans.operation_count
        ans_all.SNR = snr
        ans_all.x_tilde = x_tilde
        ans_all.benchmark = ans.benchmark
        ans_all.config = self.config
        LOGGER.info("BER/case: %s, BER/user: %s, mean: %s", ans_all.ber_of_trials, ans_all.ber_of_users, ans_all.BER)

        yield ans_all, debug_info

    @staticmethod
    def parse_MIMO(benchmark: pathlib.Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """! Parses the MIMO benchmark from the given file.

        @type benchmark: pathlib.Path
        @param benchmark: the path to the benchmark to parse.
        @rtype: tuple[np.ndarray, np.ndarray, int, int, int]
        @return: the transfer function matrix, input signals, QAM scheme, and amount of antennas and users\
            of the MIMO system.
        """

        with benchmark.open() as f:
            lines = f.readlines()

        # Parse dimensions
        M: int = int(lines[0].split()[1])  # modulation scheme QAM-M
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
        H: np.ndarray,
        x: np.ndarray,
        SNR: float,
        user_num: int,
        ant_num: int,
        M: int,
        seed: int = 0,
        is_hamming_encoding: bool = False,
    ) -> tuple[IsingModel, np.ndarray, np.ndarray]:
        """!Transforms the MIMO model into an Ising model.

        @type H: np.ndarray
        @param H: The transfer function matrix.
        @type x: np.ndarray
        @param x: the input signal.
        @type SNR: float
        @param SNR: the signal to noise ratio.
        @type user_num: int
        @param user_num: the amount of input signals.
        @type ant_num: int
        @param ant_num: the amount of output signals.
        @type M: int
        @param M: the considered QAM scheme.
        @type seed: int, optional
        @param seed: The seed for the random noise generation. Defaults to 0.
        @type is_hamming_encoding: bool, optional
        @param is_hamming_encoding: Whether to use Hamming encoding. Defaults to False.
        @rtype: tuple[IsingModel, np.ndarray, np.ndarray]
        @return: the generated Ising model, and the real version of the input, and output symbols.
        """
        is_bpsk = np.linalg.norm(np.imag(x)) == 0

        if is_bpsk:
            # BPSK scheme
            r = 1
            num_variables = np.shape(x)[0]
            num_symbols = 2 * num_variables
        else:
            if is_hamming_encoding:  # with hamming encoding
                r = int(np.sqrt(M) - 1)
            else:  # with binary encoding
                r = int(np.ceil(np.log2(np.sqrt(M))))
            num_variables = np.shape(x)[0] * 2
            num_symbols = num_variables

        if seed == 0:
            seed = int(time.time())
        np.random.seed(seed)

        # Compute the received symbols
        y = H @ x

        # Compute the amplitude of the noise on the received symbols
        power_y = np.abs(y) ** 2
        snr_real = 10 ** (SNR / 10)
        # var_noise = np.sqrt(power_y / SNR)
        n = np.random.randn(ant_num) + 1j * np.random.randn(ant_num)  # noise
        power_n = np.abs(n) ** 2
        n *= np.sqrt((power_y / snr_real) / power_n)  # take noise power into account

        assert np.all(np.isclose(np.abs(y**2 / n**2), snr_real))

        y += n

        ytilde = np.block([np.real(y), np.imag(y)])  # 2*ant_num

        Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])  # 2*ant_num x 2*user_num

        if is_hamming_encoding:  # with hamming encoding
            T = np.block([np.eye(num_symbols, num_variables) for _ in range(r)])
        else:  # with binary encoding
            T = np.block([2 ** (r - i) * np.eye(num_symbols, num_variables) for i in range(1, r + 1)])

        if is_bpsk:
            xtilde = x
        else:
            xtilde = np.block([np.real(x), np.imag(x)])

        ones_end = np.ones((num_symbols,))
        constant = ytilde.T @ ytilde - 2 * ytilde.T @ Htilde @ (
            T @ np.ones((r * num_variables,)) - (np.sqrt(M) - 1) * ones_end
        )

        bias = 2 * (ytilde - Htilde @ (T @ np.ones((r * num_variables,)) - (np.sqrt(M) - 1) * ones_end))
        bias = bias.T @ Htilde @ T
        coupling = -2 * T.T @ Htilde.T @ Htilde @ T
        constant -= np.sum(np.diag(coupling)) / 2

        coupling = np.triu(coupling, k=1)
        return IsingModel(coupling, bias, constant, name=f"MIMO_{SNR}"), xtilde, y
