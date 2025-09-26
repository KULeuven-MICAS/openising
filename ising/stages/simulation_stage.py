from ising.stages import LOGGER, TOP
from abc import ABCMeta
from typing import Any
import numpy as np
import datetime
import tqdm
import pathlib
import os
import multiprocessing

from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel
from ising.solvers.Gurobi import Gurobi
from ising.utils.flow import parse_hyperparameters
from ising.solvers.BRIM import BRIM
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.SA import SASolver
from ising.solvers.DSA import DSASolver
from ising.solvers.inSitu_SA import InSituSASolver
from ising.solvers.CIM import CIMSolver
from ising.solvers.Multiplicative import Multiplicative


class SimulationStage(Stage):
    """! Stage to simulate the Ising model and evaluate its Hamiltonian."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        config: Any,
        ising_model: IsingModel,
        best_found: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.gen_logfile = self.config.gen_logfile if hasattr(self.config, "gen_logfile") else False
        self.ising_model = ising_model
        self.best_found = best_found if best_found is not None else float("inf")
        self.benchmark_abbreviation = self.config.benchmark.split("/")[-1].split(".")[0]
        if self.config.logfile_discrimination is not None:
            self.logfile_discrimination = self.config.logfile_discrimination
        else:
            self.logfile_discrimination = ""
        if "run_id" in self.kwargs:
            self.run_id = self.kwargs["run_id"]

    def run(self) -> Any:
        """! Simulate the Ising model and evaluate its Hamiltonian."""

        problem_type = self.config.problem_type
        nb_cores = self.config.nb_cores
        logpath = TOP / f"ising/outputs/{problem_type}/logs"
        LOGGER.debug("Logpath: " + str(logpath))
        if not logpath.exists():
            logpath.mkdir(parents=True, exist_ok=True)

        if bool(int(self.config.use_gurobi)):
            gurobi_log = logpath / f"Gurobi_{self.benchmark_abbreviation}.log"
            Gurobi().solve(model=self.ising_model, file=gurobi_log)

        nb_runs = int(self.config.nb_runs)  # Number of trails
        if self.config.use_multiprocessing:
            runs_per_thread = nb_runs // nb_cores

        start_time = datetime.datetime.now()

        optim_state_collect = {solver: [] for solver in self.config.solvers}
        optim_energy_collect = {solver: [] for solver in self.config.solvers}
        comp_time_collect = {solver: [] for solver in self.config.solvers}
        operation_count = {solver: -1 for solver in self.config.solvers}
        logfile_collect = []
        if self.config.use_multiprocessing:
            runs_over = nb_runs - runs_per_thread * nb_cores
            tasks = [
                (runs_per_thread + 1, logpath, i, i * (runs_per_thread + 1))
                if i < runs_over
                else (
                    runs_per_thread,
                    logpath,
                    i,
                    runs_over * (runs_per_thread + 1) + (i - runs_over) * runs_per_thread,
                )
                for i in range(nb_cores)
            ]
            with multiprocessing.Pool(nb_cores, initializer=os.nice, initargs=(1,)) as pool:
                results = pool.starmap(self.partial_runs, tasks)
        else:
            results = self.partial_runs(nb_runs, logpath, 0)
            results = [results]
        end_time = datetime.datetime.now()
        LOGGER.info(f"Simulation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.info(f"Total simulation time: {end_time - start_time}")

        for res in results:
            for solver in self.config.solvers:
                optim_state_collect[solver] += res[0][solver]
                optim_energy_collect[solver] += res[1][solver]
                comp_time_collect[solver] += res[2][solver]
                operation_count[solver] = res[3][solver]
            logfile_collect += res[4]

        ans = Ans(
            benchmark=self.benchmark_abbreviation,
            ising_model=self.ising_model,
            config=self.config,
            best_found=self.best_found,
            states=optim_state_collect,
            energies=optim_energy_collect,
            computation_time=comp_time_collect,
            operation_count=operation_count,
            logfiles=logfile_collect,
        )
        debug_info = Ans()  # Placeholder for debug information, if needed

        yield ans, debug_info

    def partial_runs(self, nb_runs: int, logpath: pathlib.Path, initialization_seed: int, start_run_id: int = 0):
        start_time = datetime.datetime.now()
        LOGGER.info(f"Simulation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        num_iter = self.config.iter_list
        hyperparameters = parse_hyperparameters(self.config, num_iter)

        optim_state_collect = {solver:[] for solver in self.config.solvers}
        optim_energy_collect = {solver:[] for solver in self.config.solvers}
        comp_time_collect = {solver:[] for solver in self.config.solvers}
        nb_operations_collect = {solver: -1 for solver in self.config.solvers}
        logfile_collect = []
        pbar = tqdm.tqdm(range(nb_runs), ascii="░▒█", desc=f"Running trials of thread {os.getpid()}")
        for trail_id in pbar:
            # Set the seed for flipping mechanism
            hyperparameters["seed"] = trail_id + 1 + int(self.config.seed + initialization_seed)

            self.kwargs["config"] = self.config
            self.kwargs["ising_model"] = self.ising_model
            self.kwargs["trail_id"] = trail_id
            if len(self.list_of_callables) >= 1:
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
                initial_state, _ = sub_stage.run()
            else:
                initial_state = np.random.uniform(-1, 1, (self.ising_model.num_variables,))

            for solver in self.config.solvers:
                if self.gen_logfile and self.benchmark_abbreviation != "MIMO":
                    logfile = (
                        logpath
                        / f"{solver}_{self.benchmark_abbreviation}_nbiter{num_iter}_run{trail_id +\
                                                                                         start_run_id}_{self.logfile_discrimination}.log"
                    )
                else:
                    logfile = None

                optim_state, optim_energy, computation_time, nb_operations = self.run_solver(
                    solver, num_iter, initial_state, self.ising_model, logfile, **hyperparameters
                )

                optim_state_collect[solver].append(optim_state)
                optim_energy_collect[solver].append(optim_energy)
                comp_time_collect[solver].append(computation_time)
                nb_operations_collect[solver] = nb_operations
                logfile_collect.append(logfile)
            pbar.set_description(
                f"Running trails of thread {os.getpid()} [#{trail_id + 1}, energy: {optim_energy:.2f}]"
            )
        end_time = datetime.datetime.now()
        LOGGER.info(f"Simulation of thread {os.getpid()} finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.info(f"Thread {os.getpid()} simulation time: {end_time - start_time}")
        return optim_state_collect, optim_energy_collect, comp_time_collect, nb_operations_collect, logfile_collect

    def run_solver(
        self,
        solver: str,
        num_iter: int,
        s_init: np.ndarray,
        model: IsingModel,
        logfile: pathlib.Path | None = None,
        **hyperparameters,
    ) -> tuple[np.ndarray, float]:
        """! Solves the given problem with the specified solver.

        @param solver: The solver to use
        @param num_iter: The number of iterations to run the solver
        @param s_init: Initial state for the solver
        @param model: The Ising model to use for the solver
        @param logfile: Path to the logfile to store data. Defaults to None.

        @return optim_state: optimal state of the specified solver.
        @return optim_energy: optimal energy of the specified solver.
        """
        optim_state = np.zeros((model.num_variables,))
        optim_energy = None
        solvers = {
            "BRIM": (
                BRIM().solve,
                [
                    "dtBRIM",
                    "capacitance",
                    "stop_criterion",
                    "initial_temp_cont",
                    "end_temp_cont",
                    "seed",
                    "coupling_annealing",
                ],
            ),
            "Multiplicative": (
                Multiplicative().solve,
                [
                    "dtMult",
                    "initial_temp_cont",
                    "end_temp_cont",
                    "seed",
                    "capacitance",
                    "resistance",
                    "nb_flipping",
                    "cluster_threshold",
                    "init_cluster_size",
                    "end_cluster_size",
                    "cluster_choice",
                    "pseudo_length"
                ],
            ),
            "inSituSA": (InSituSASolver().solve, ["initial_temp", "cooling_rate", "nb_flips", "seed"]),
            "SA": (SASolver().solve, ["initial_temp", "cooling_rate", "seed"]),
            "DSA": (DSASolver().solve, ["initial_temp", "cooling_rate", "seed"]),
            "SCA": (SCA().solve, ["initial_temp", "cooling_rate", "q", "r_q", "seed"]),
            "bSB": (ballisticSB().solve, ["c0", "dtSB", "a0"]),
            "dSB": (discreteSB().solve, ["c0", "dtSB", "a0"]),
            "CIM": (CIMSolver().solve, ["dtCIM", "zeta", "seed"]),
        }
        if solver in solvers:
            func, params = solvers[solver]
            chosen_hyperparameters = {key: hyperparameters[key] for key in params if key in hyperparameters}
            optim_state: np.ndarray
            optim_energy: float | None
            optim_state, optim_energy, computation_time = func(
                model=model,
                initial_state=s_init,
                num_iterations=num_iter,
                file=logfile,
                **chosen_hyperparameters,
            )
        else:
            LOGGER.error(f"Solver {solver} is not implemented.")
            raise NotImplementedError(f"Solver {solver} is not implemented.")
        return optim_state, optim_energy, computation_time


class Ans(metaclass=ABCMeta):
    """! Abstract class for the answer of the simulation stage."""

    def __init__(self, **kwargs: Any):
        """! Initializes the answer with the given parameters."""
        object.__setattr__(self, "kwargs", kwargs)
        object.__setattr__(self, "_attributes", {})
        # Set initial attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """! Sets an attribute of the answer."""
        self._attributes[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """! Gets an attribute of the answer."""
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
