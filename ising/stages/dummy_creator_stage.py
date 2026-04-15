from ising.stages import LOGGER
from typing import Any
import networkx as nx
import numpy as np
from argparse import Namespace

from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel
from ising.generators.TSP import TSP
from ising.stages.qkp_parser_stage import QKPParserStage


class DummyCreatorStage(Stage):
    """! Stage to create a dummy Ising model for testing purposes.
    To create a dummy model, the problem type and size must be specified in the yaml configuration.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, config: Any, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.problem_type = config.problem_type

    def run(self) -> Any:
        """! Creates a dummy Ising model."""

        dummy_creator = self.config.dummy_creator if hasattr(self.config, "dummy_creator") else False
        if dummy_creator:
            LOGGER.info(f"Creating a dummy {self.problem_type} model.")
            seed = self.config.dummy_seed

            if self.problem_type == "Maxcut":
                N = self.config.dummy_size
                LOGGER.info(f"size: {N}, seed: {seed}")
                nb_bits = round(self.config.quantization_precision) if hasattr(self.config, "quantization_precision")\
                    else 2
                dummy_dict = self.generate_dummy_maxcut(N, nb_bits, seed)
            elif self.problem_type in ["TSP", "ATSP"]:
                N = self.config.dummy_size
                weight_constant = (
                    self.config.dummy_weight_constant if hasattr(self.config, "dummy_weight_constant") else 0.0
                )
                if not hasattr(self.config, "weight_constant"):
                    LOGGER.warning("No weight_constant provided in config, using default value of 1.0.")
                    weight_constant = 1.0
                LOGGER.info(f"size: {N}, seed: {seed}, weight_constant: {weight_constant}")
                if self.problem_type == "TSP":
                    dummy_dict = self.generate_dummy_tsp(N, seed, weight_constant=weight_constant)
                else:
                    dummy_dict = self.generate_dummy_atsp(N, seed, weight_constant=weight_constant)
            elif self.problem_type == "MIMO":
                dummy_qam = self.config.dummy_qam if hasattr(self.config, "dummy_qam") else 4
                dummy_snr = self.config.dummy_snr if hasattr(self.config, "dummy_snr") else 10
                dummy_spacing = self.config.dummy_spacing if hasattr(self.config, "dummy_spacing") else 1.0
                user_num = self.config.dummy_user_num if hasattr(self.config, "dummy_user_num") else 4
                ant_num = self.config.dummy_ant_num if hasattr(self.config, "dummy_ant_num") else 4
                dummy_case_num = self.config.dummy_case_num if hasattr(self.config, "dummy_case_num") else 10
                LOGGER.info(
                    f"QAM: {dummy_qam}, SNR: {dummy_snr}, user_num: {user_num}, ant_num: {ant_num}, "
                    f"seed: {seed}, case_num: {dummy_case_num}"
                )
                dummy_dict = self.generate_dummy_mimo(
                    user_num=user_num,
                    ant_num=ant_num,
                    M=dummy_qam,
                    SNR=dummy_snr,
                    antenna_spacing=dummy_spacing,
                    seed=seed,
                    dummy_case_num=dummy_case_num,
                )
            elif self.problem_type == "Knapsack":
                N = self.config.dummy_size
                density = self.config.dummy_density if hasattr(self.config, "dummy_density") else 1
                penalty_value = self.config.dummy_penalty_value if hasattr(self.config, "dummy_penalty_value") else 1.0
                bit_width = self.config.dummy_bit_width if hasattr(self.config, "dummy_bit_width") else 16
                LOGGER.info(
                    f"size: {N}, density: {density}, penalty_value: {penalty_value}, "
                    f"bit_width: {bit_width}, seed: {seed}"
                )
                dummy_dict = self.generate_dummy_knapsack(
                    size=N, dns=density, penalty_value=penalty_value, bit_width=bit_width
                )
            else:
                LOGGER.error(f"Dummy creator for {self.problem_type} is not supported.")
                raise NotImplementedError(f"Dummy creator for {self.problem_type} is not implemented.")

            self.kwargs["config"] = self.config
            self.kwargs["dummy_dict"] = dummy_dict
            self.kwargs["best_found"] = None
        else:
            LOGGER.info("Dummy creator is disabled, skipping dummy model creation.")
            self.kwargs["config"] = self.config

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    @staticmethod
    def generate_dummy_maxcut(N: int, nb_bits: int = 2, seed: int = 0) -> dict:
        """! Generates a random Max Cut Ising model.

        @type N: int
        @param N: Number of nodes in the graph.
        @type nb_bits: int
        @param nb_bits: number of bits to represent the weights.
        @type seed: int
        @param seed: Random seed for reproducibility.
        @rtype: dict
        @return: dict containing graph and IsingModel representing the Max Cut problem.
        """

        np.random.seed(seed)
        name = f"DummyMaxCut_N{N}_seed{seed}"
        J = np.random.choice(np.arange(int(-(2 ** (nb_bits - 1)-1)), int(2 ** (nb_bits - 1))), (N, N))

        # Map the J matrix to a graph
        graph = nx.Graph(name=name)
        graph.add_nodes_from(range(1, N + 1))  # Nodes are 1-indexed in the graph
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    graph.add_edge(i + 1, j + 1, weight=-J[i, j] * 2)

        J = np.triu(J, k=1)  # Keep only upper triangle
        h = np.zeros((N,))  # No external field
        c = np.sum(J)  # Constant term for the Max Cut problem
        ising_model = IsingModel(J, h, c, name=name)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_tsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> dict:
        """! Generates a random TSP Ising model.
        @type N: int
        @param N: Number of cities (nodes) in the TSP problem.
        @type seed: int
        @param seed: Random seed for reproducibility.
        @type weight_constant: float
        @param weight_constant: Constant to scale the weights in the TSP problem.
        @rtype: dict
        @return: dict containing graph and IsingModel representing the TSP problem.
        """

        np.random.seed(seed)
        name = f"DummyTSP_N{N}_seed{seed}"
        W = np.random.choice(10, (N, N))
        W = (W + W.T) / 2  # Make it symmetric

        graph = nx.DiGraph(name=name)
        graph.add_nodes_from(range(1, N + 1))
        for i in range(N):
            for j in range(N):
                if i != j:
                    if W[i, j] != 0:
                        graph.add_edge(i + 1, j + 1, weight=W[i, j])

        ising_model = TSP(graph, weight_constant=weight_constant)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
            "weight_constant": weight_constant,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_atsp(N: int, seed: int = 0, weight_constant: float = 1.0) -> dict:
        """! Generates a random ATSP Ising model.
        @type N: int
        @param N: Number of cities (nodes) in the ATSP problem.
        @type seed: int
        @param seed: Random seed for reproducibility.
        @type weight_constant: float
        @param weight_constant: Constant to scale the weights in the ATSP problem.
        @rtype: dict
        @return: dict containing graph and IsingModel representing the ATSP problem.
        """

        np.random.seed(seed)
        name = f"DummyATSP_N{N}_seed{seed}"
        W = np.random.choice(10, (N, N))

        graph = nx.DiGraph(name=name)
        graph.add_nodes_from(range(1, N + 1))
        for i in range(N):
            for j in range(N):
                if i != j:
                    if W[i, j] != 0:
                        graph.add_edge(i + 1, j + 1, weight=W[i, j])

        ising_model = TSP(graph, weight_constant=weight_constant)

        dummy_dict: dict = {
            "ising_model": ising_model,
            "graph": graph,
            "N": N,
            "seed": seed,
            "weight_constant": weight_constant,
        }

        return dummy_dict

    @staticmethod
    def generate_dummy_mimo(
        user_num: int,
        ant_num: int,
        M: int,
        SNR: int,
        antenna_spacing: float = 1.0,
        seed: int = 0,
        dummy_case_num: int = 10,
    ) -> dict:
        """!Generates a MU-MIMO model using section IV-A of [this paper](https://arxiv.org/pdf/2002.02750).
        This is consecutively transformed into an Ising model.

        @type user_num: int
        @param ant_num: The amount of users.
        @type ant_num: int
        @param user_num: The amount of antennas at the Base Station.
        @type M: int
        @param M: the considered QAM scheme.
        @type SNR: int
        @param SNR: the Signal-to-Noise Ratio.
        @type antenna_spacing: float, optional
        @param antenna_spacing: The spacing between antennas in wavelengths. Defaults to 1.0.
        @type seed: int, optional
        @param seed: The seed for the random number generator. Defaults to 1.
        @type dummy_case_num: int, optional
        @param dummy_case_num: The number of dummy trails to generate. Defaults to 10.
        @rtype: dict
        @return: dict containing IsingModel representing the MIMO problem.
        """
        np.random.seed(seed)

        # modulation scheme must be a power of 2
        assert (M & (M - 1) == 0) and M != 0, f"Modulation {M} must be a power of 2"
        assert M == 2 or np.sqrt(M).is_integer(), f"Modulation {M} must be a square of an integer"

        if M == 2:
            # BPSK scheme
            symbols = np.array([-1, 1])
            r = 1
        else:
            r = int(np.ceil(np.log2(np.sqrt(M))))
            symbols = np.concatenate(
                ([-np.sqrt(M) + i for i in range(1, 2 + 2 * r, 2)], [np.sqrt(M) - i for i in range(1, 2 + 2 * r, 2)])
            )

        mean_phi = 120 * np.pi / 180 * (np.random.random((user_num,)) - 0.5)
        mean_phi.sort()
        sigma_phi = np.array([10 * np.pi / 180] * user_num)

        # H = np.random.random((ant_num, user_num)) + 1j*np.random.random((ant_num, user_num*np.pi/180))
        H = np.zeros((ant_num, user_num), dtype=np.complex128)
        for i in range(user_num):
            C = np.zeros((ant_num, ant_num), dtype=np.complex128)
            phi = mean_phi[i]
            sigma = sigma_phi[i]
            for m in range(ant_num):
                for n in range(ant_num):
                    d = antenna_spacing * (m - n)
                    C[m, n] = np.exp(2 * np.pi * 1j * d * np.sin(phi)) * np.exp(
                        -((sigma) ** 2) / 2 * (2 * np.pi * d * np.cos(phi)) ** 2
                    )
            D, V = np.linalg.eig(C)
            hu = (
                V
                @ np.diag(D) ** 0.5
                @ V.conj().T
                @ (np.random.normal(0, 1, (ant_num,)) + 1j * np.random.normal(0, 1, (ant_num,)))
            )
            H[:, i] = hu

        if M == 2:
            x_collect = np.random.choice(symbols, size=(user_num, dummy_case_num))
        else:
            x_collect = np.random.choice(symbols, size=(user_num, dummy_case_num)) + 1j * np.random.choice(
                symbols, size=(user_num, dummy_case_num)
            )

        dummy_dict: dict = {
            "H": H,
            "x_collect": x_collect,
            "user_num": user_num,
            "ant_num": ant_num,
            "M": M,
            "SNR": SNR,
            "seed": seed,
        }
        return dummy_dict

    def generate_dummy_knapsack(size: int, dens: int, penalty_value: float = 1.0, bit_width: int = 16) -> dict:
        """! Generates a dummy knapsack problem instance.

        @type size: int
        @param size: the number of items.
        @type dens: int
        @param dens: the density of the problem.
        @type penalty_value: float, optional
        @param penalty_value: the penalty value for the constraint. Defaults to 1.0.
        @type bit_width: int, optional
        @param bit_width: the number of bits to represent the profits and weights. Defaults to 16.
        @rtype: dict
        @return: dictionary containing the generated IsingModel and problem parameters.
        """
        max_number = int(2**bit_width)
        profit = np.triu(
            np.random.choice(
                max_number + 1, size=(size, size), p=[1, -dens / 100] + [dens / (dens * max_number)] * (max_number - 1)
            )
        )
        profit = profit + profit.T
        weights = np.random.randint(1, max_number, size=(size,))
        capacity = np.random.randint(np.min(weights) * 2, np.sum(weights) - np.min(weights), size=(1,))[0]

        ising_model: IsingModel = QKPParserStage([StageCallable], config=Namespace()).knapsack_to_ising(
            profit, capacity, weights, penalty_value
        )

        dummy_dict: dict = {
            "ising_model": ising_model,
            "N": size,
            "density": dens,
            "penalty_value": penalty_value,
            "bit_width": bit_width,
        }

        return dummy_dict
