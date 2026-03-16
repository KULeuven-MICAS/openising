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
        if not config.dummy_creator:
            self.benchmark_abbreviation = self.config.benchmark.split("/")[-1].split(".")[0]
        else:
            self.benchmark_abbreviation = f"dummy_{self.config.problem_type}"
        if self.config.logfile_discrimination != "None":
            self.logfile_discrimination = "_" + self.config.logfile_discrimination
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
        initialization_state_collect = []
        logfile_collect = []
        if self.config.use_multiprocessing:
            runs_over = nb_runs - runs_per_thread * nb_cores
            tasks = [
                (runs_per_thread + 1, logpath, self.config.initialization_seed + i, i * (runs_per_thread + 1))
                if i < runs_over
                else (
                    runs_per_thread,
                    logpath,
                    self.config.initialization_seed + i,
                    runs_over * (runs_per_thread + 1) + (i - runs_over) * runs_per_thread,
                )
                for i in range(nb_cores)
            ]
            with multiprocessing.Pool(nb_cores, initializer=os.nice, initargs=(5,)) as pool:
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
            initialization_state_collect += res[5]

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
            initialization_states = initialization_state_collect,
        )
        debug_info = Ans()  # Placeholder for debug information, if needed

        yield ans, debug_info

    def partial_runs(self, nb_runs: int, logpath: pathlib.Path, initialization_seed: int, start_run_id: int = 0):
        start_time = datetime.datetime.now()
        LOGGER.info(f"Simulation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        hyperparameters = parse_hyperparameters(self.config)

        optim_state_collect = {solver: [] for solver in self.config.solvers}
        optim_energy_collect = {solver: [] for solver in self.config.solvers}
        comp_time_collect = {solver: [] for solver in self.config.solvers}
        nb_operations_collect = {solver: -1 for solver in self.config.solvers}
        initialization_state_collect = []
        logfile_collect = []
        pbar = tqdm.tqdm(range(nb_runs), ascii="░▒█", desc=f"Running trials of thread {os.getpid()}")
        for trail_id in pbar:
            # Set the seed for flipping mechanism
            if self.config.seed > 0:
                hyperparameters["seed"] = start_run_id + trail_id + int(self.config.seed)
            self.kwargs["config"] = self.config
            self.kwargs["ising_model"] = self.ising_model
            self.kwargs["trail_id"] = trail_id
            if len(self.list_of_callables) >= 1:
                self.kwargs["initialization_seed"] = initialization_seed + start_run_id + trail_id
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
                initial_state, _ = sub_stage.run()
            else:
                initial_state = np.random.uniform(-1, 1, (self.ising_model.num_variables,))
            for solver in self.config.solvers:
                if self.gen_logfile and self.benchmark_abbreviation != "MIMO":
                    logfile = (
                        logpath
                        / f"{solver}_{self.benchmark_abbreviation}_run{trail_id + start_run_id}{
                            self.logfile_discrimination
                        }.log"
                    )
                else:
                    logfile = None

                optim_state, optim_energy, computation_time, nb_operations = self.run_solver(
                    solver, initial_state, self.ising_model, logfile, **hyperparameters
                )
                optim_state_collect[solver].append(optim_state)
                optim_energy_collect[solver].append(optim_energy)
                comp_time_collect[solver].append(computation_time)
                nb_operations_collect[solver] = nb_operations
                logfile_collect.append(logfile)
            initialization_state_collect.append(initial_state)

            pbar.set_description(
                f"Running trails of thread {os.getpid()} [#{trail_id + 1}, energy: {optim_energy:.2f}]"
            )
        end_time = datetime.datetime.now()
        LOGGER.info(f"Simulation of thread {os.getpid()} finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.info(f"Thread {os.getpid()} simulation time: {end_time - start_time}")
        return (
            optim_state_collect,
            optim_energy_collect,
            comp_time_collect,
            nb_operations_collect,
            logfile_collect,
            initialization_state_collect,
        )

    def run_solver(
        self,
        solver: str,
        s_init: np.ndarray,
        model: IsingModel,
        logfile: pathlib.Path | None = None,
        **hyperparameters,
    ) -> tuple[np.ndarray, float]:
        """! Solves the given problem with the specified solver.

        @param solver: The solver to use
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
                ["dtBRIM", "capacitance", "resistance", "stop_criterion", "seed", "coupling_annealing", "do_flipping"],
            ),
            "Multiplicative": (
                Multiplicative().solve,
                [
                    "dtMult",
                    "seed",
                    "current",
                    "capacitance",
                    "nb_flipping",
                    "cluster_threshold",
                    "init_cluster_size",
                    "end_cluster_size",
                    "exponent",
                    "cluster_choice",
                    "ode_choice",
                    "stop_criterion",
                    "accumulation_delay",
                    "broadcast_delay",
                    "delay_offset",
                    "sigma_J",
                    "sigma_C",
                ],
            ),
            "inSituSA": (
                InSituSASolver().solve,
                ["initial_temp_inSituSA", "cooling_rate_inSituSA", "nb_flips", "seed"],
            ),
            "SA": (SASolver().solve, ["initial_temp", "cooling_rate_SA", "seed"]),
            "DSA": (DSASolver().solve, ["initial_temp", "cooling_rate_SA", "seed"]),
            "SCA": (SCA().solve, ["initial_temp", "cooling_rate_SCA", "q", "r_q", "seed"]),
            "bSB": (ballisticSB().solve, ["c0", "dtSB", "a0", "seed"]),
            "dSB": (discreteSB().solve, ["c0", "dtSB", "a0", "seed"]),
            "CIM": (CIMSolver().solve, ["dtCIM", "zeta", "seed"]),
        }
        if solver in solvers:
            func, params = solvers[solver]
            chosen_hyperparameters = {key: hyperparameters[key] for key in params if key in hyperparameters}
            optim_state: np.ndarray
            optim_energy: float | None
            optim_state, optim_energy, computation_time, operation_count = func(
                model=model,
                initial_state=s_init,
                num_iterations=hyperparameters[
                    "num_iterations_" + solver if ((solver != "bSB") and (solver != "dSB")) else "num_iterations_SB"
                ],
                file=logfile,
                **chosen_hyperparameters,
            )
        else:
            LOGGER.error(f"Solver {solver} is not implemented.")
            raise NotImplementedError(f"Solver {solver} is not implemented.")
        return optim_state, optim_energy, computation_time, operation_count


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
        try:
            attrs = object.__getattribute__(self, "_attributes")
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in attrs:
            return attrs[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
