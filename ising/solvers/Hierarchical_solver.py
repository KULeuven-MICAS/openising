import pathlib
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from ising.solvers.base import SolverBase
from ising.stages.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger
from ising.utils.partitioning_schemes import (
    make_louvain_partitioning,
    make_modularity_partitioning,
    make_random_partitioning,
    make_spectral_partitioning,
)


@dataclass
class Subproblem:
    """A local model, its mappings, and its current solution.

    Both index arrays follow local-node order. ``original_nodes`` indexes the
    full model; ``upper_nodes`` indexes the corresponding meta-nodes.
    """

    model: IsingModel
    original_nodes: np.ndarray
    upper_nodes: np.ndarray
    state: np.ndarray = field(init=False)

    def initialize(self, original_state: np.ndarray) -> None:
        self.state = original_state[self.original_nodes].copy()

    def write_state(self, original_state: np.ndarray) -> None:
        original_state[self.original_nodes] = self.state

    def vote_into(self, upper_state: np.ndarray) -> None:
        """Update represented meta-nodes, retaining their previous spin on ties.

        @type upper_state: np.ndarray
        @param upper_state: the state of the upper model that is updated with the subproblems state.
        """
        for upper_node in np.unique(self.upper_nodes):
            vote = self.state[self.upper_nodes == upper_node].sum()
            if vote != 0:
                upper_state[upper_node] = np.sign(vote)


class HierarchicalSolver(SolverBase):
    """Own the hierarchy, its state, and the upper/lower sweep schedule.

    The core solver is an instance exposing ``solve``. Its result must start
    with state and energy; additional metrics are ignored. This solver returns
    ``(state, energy)``, preserving its existing runtime return contract.
    """

    def __init__(self):
        super().__init__()
        self.name = "Hierarchical_solver"
        self.original_model: IsingModel | None = None
        self.upper_model: IsingModel | None = None
        self.subproblems: dict[int, Subproblem] = {}
        self.original_coupling: np.ndarray | None = None
        self.original_to_upper: np.ndarray | None = None
        self.upper_state: np.ndarray | None = None

    def solve(
        self,
        model: IsingModel,
        initial_state: np.ndarray,
        nb_sweeps: int,
        core_solver: SolverBase,
        partitioning_technique: str,
        nb_partitions: int,
        nb_meta_nodes: int | None,
        core_solver_args: dict,
        file_multi_core: pathlib.Path | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Performs the hierarchical multi-core solver. The model is partitioned into lower
        subproblems and an upper model of meta-nodes. Each sweep optimizes the upper model,
        then solves all lower subproblems using the same upper state. Votes update the
        upper state only after all lower subproblems have finished.

        ``nb_meta_nodes=None`` uses the size of the first partition.
        ``file`` is retained for compatibility; hierarchy logging is not yet
        implemented. Core-solver options are forwarded via ``core_solver_args``.

        @type model: IsingModel
        @param model: the original Ising model to solve.
        @type initial_state: np.ndarray
        @param initial_state: initial spins of the original model, containing only -1 and +1.
        @type nb_sweeps: int
        @param nb_sweeps: the number of sweeps over the upper model and all lower subproblems.
        @type core_solver: SolverBase
        @param core_solver: solver instance used for both the upper model and lower subproblems.
            Its solve method returns state, energy, time, operation count, and iteration count.
        @type partitioning_technique: str
        @param partitioning_technique: partitioning method: "random", "modularity", or "spectral".
        @type nb_partitions: int
        @param nb_partitions: the number of lower subproblems to create.
        @type nb_meta_nodes: int or None
        @param nb_meta_nodes: the total number of meta-nodes in the upper model.
            If None, use the number of original nodes in the first partition.
        @type core_solver_args: dict
        @param core_solver_args: keyword arguments forwarded to each core-solver call.
        @type file: pathlib.Path or None
        @param file: path to the HDF5 logfile. If None, logging is disabled. Defaults to None.
        @rtype: tuple[np.ndarray, float, float, int, int]
        @return: the final state in original-node order, its energy on the original model,
            total time, operation count, and number of sweeps.
        """
        if nb_sweeps < 0:
            raise ValueError("Number of sweeps cannot be negative")

        partitioners = {
            "random": make_random_partitioning,
            "modularity": make_modularity_partitioning,
            "spectral": make_spectral_partitioning,
            "louvain": make_louvain_partitioning,
        }
        if partitioning_technique not in partitioners:
            raise ValueError(f"Partitioning technique {partitioning_technique} does not exist.")
        partitioning = partitioners[partitioning_technique](model, nb_partitions)
        self.make_hierarchy(model, partitioning, nb_meta_nodes)
        self.initialize(initial_state)

        schema = {
            "time": np.float32,
            "energy": np.float32,
            "state": (np.int8, (model.num_variables,)),
            "energy_partitions": (np.float32, (nb_partitions)),
            "time_partitions": (np.float32, (nb_partitions)),
            "operations_partitions": (int, (nb_partitions,)),
            "operations_per_iteration": (np.float32, (nb_partitions)),
        }

        singleton_case = (
            self.upper_model.num_variables == model.num_variables
        )  # case when every node in the original model is represented by a meta-node
        time = 0
        operations = 0
        with HDF5Logger(file_multi_core, schema) as logger:
            self.log_metadata(
                logger, initial_state, model, nb_sweeps, nb_partitions=nb_partitions, nb_meta_nodes=nb_meta_nodes
            )

            for _ in range(nb_sweeps):
                self.upper_state, en, time_upper, operations_upper, num_iterations = core_solver(
                    model=self.upper_model, initial_state=self.upper_state, **core_solver_args
                )
                if singleton_case:
                    # Every original node is represented individually at the upper level.
                    expanded = self.expand_upper()
                    for subproblem in self.subproblems.values():
                        subproblem.initialize(expanded)
                    continue
                times = []
                energies = []
                operations_part = []
                op_it = []
                for subproblem in self.subproblems.values():
                    local_model = self.get_influence(subproblem)
                    # An unbiased singleton has constant energy; retain its spin.
                    if local_model.num_variables != 1 or local_model.h[0] != 0:
                        subproblem.state, en, time, operations, num_iterations = core_solver(
                            model=local_model, initial_state=subproblem.state, **core_solver_args
                        )
                        times.append(time)
                        energies.append(en)
                        operations_part.append(operations)
                        op_it.append(operations / num_iterations)
                for subproblem in self.subproblems.values():
                    subproblem.vote_into(self.upper_state)
                time += time_upper + np.max(times)
                operations += (
                    operations_upper + np.max(operations_part) + 2 * model.num_variables**2
                )  # influence calculation
                if logger.filename is not None:
                    final_state = self.assemble_state()
                    energy = model.evaluate(final_state)
                    logger.log(
                        time_partitions=time,
                        operations_partitions=operations_part,
                        operations_per_iteration=op_it,
                        energy_partitions=energies,
                        time=time,
                        state=final_state,
                        energy=energy,
                    )
            final_state = self.assemble_state()
            energy = model.evaluate(final_state)
            if logger.filename is not None:
                logger.write_metadata(solution_state=final_state, solution_energy=energy, total_time=time)
        return (final_state, energy, time, operations, nb_sweeps)

    def make_hierarchy(self, original_model: IsingModel, partitioning: np.ndarray, nb_meta_nodes: int | None) -> None:
        """Build local models and the mean-coupling upper model for a new solve.

        @type original_model: IsingModel
        @param original_model: the original model of which the hierarchy is made
        @type partitioning: np.ndarray
        @param partitioning: the partitioning of the original model
        @type nb_meta_nodes: int, None
        @param nb_meta_nodes: amount of meta nodes used in the upper model. When set to None, the amount is set to the \
            size of the first partition.
        """
        part_ids = np.unique(partitioning)
        num_upper_nodes = int(np.count_nonzero(partitioning == part_ids[0])) if nb_meta_nodes is None else nb_meta_nodes
        upper_nodes_part, remainder = divmod(num_upper_nodes, len(part_ids))
        if upper_nodes_part < 1:
            raise ValueError("Each subproblem needs at least one upper node")

        subproblems = {}
        original_to_upper = np.empty(original_model.num_variables, dtype=int)
        upper_index = 0
        for part_id in part_ids:
            original_ids = np.flatnonzero(partitioning == part_id)
            group_count = upper_nodes_part + int(remainder > 0)
            if group_count > len(original_ids):
                raise ValueError("More upper nodes requested than local nodes")
            lower_model = IsingModel(
                np.triu(original_model.J[np.ix_(original_ids, original_ids)], k=1),
                original_model.h[original_ids],
            )
            lower_labels = self.make_meta_nodes(lower_model, group_count)
            groups, local_groups = np.unique(lower_labels, return_inverse=True)
            if lower_labels.shape != (lower_model.num_variables,) or len(groups) != group_count:
                raise ValueError("Meta-node grouping returned an unexpected shape or group count")
            upper_nodes = upper_index + local_groups
            subproblems[part_id] = Subproblem(lower_model, original_ids, upper_nodes)
            original_to_upper[original_ids] = upper_nodes
            upper_index += group_count
            remainder -= 1

        original_coupling = triu_to_symm(original_model.J)
        upper_members = [np.flatnonzero(original_to_upper == i) for i in range(num_upper_nodes)]
        upper_J = np.zeros((num_upper_nodes, num_upper_nodes))
        upper_h = np.zeros(num_upper_nodes)

        # Upper h is just the mean of the biases of the nodes in that meta node
        for i, members_i in enumerate(upper_members):
            upper_h[i] = original_model.h[members_i].mean()

        pairs_by_shape = defaultdict(list)
        for i in range(num_upper_nodes):
            for j in range(i + 1, num_upper_nodes):
                pairs_by_shape[len(upper_members[i]), len(upper_members[j])].append((i, j))

        for pairs in pairs_by_shape.values():
            i, j = np.array(pairs).T
            rows = np.array([upper_members[k] for k in i])
            cols = np.array([upper_members[k] for k in j])
            upper_J[i, j] = original_coupling[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))

        self.original_model = original_model
        self.original_coupling = original_coupling
        self.original_to_upper = original_to_upper
        self.upper_model = IsingModel(upper_J, upper_h)
        self.subproblems = subproblems
        self.upper_state = None

    def initialize(self, initial_state: np.ndarray) -> None:
        """Initialize local spins and randomly initialize the upper spins.

        @type initial_state: np.ndarray
        @param initial_state: the given initial state.
        """
        self.upper_state = np.random.choice([-1, 1], self.upper_model.num_variables)
        for subproblem in self.subproblems.values():
            subproblem.initialize(initial_state)
            subproblem.vote_into(self.upper_state)

    def expand_upper(self) -> np.ndarray:
        return self.upper_state[self.original_to_upper]

    def assemble_state(self) -> np.ndarray:
        """Assemble the subproblem states into a final state for the original problem.

        @rtype: np.ndarray
        @return: the final state in the shape of the original problem.
        """
        result = np.empty(self.original_model.num_variables, dtype=np.int8)
        for subproblem in self.subproblems.values():
            subproblem.write_state(result)
        return result

    def get_influence(self, subproblem: Subproblem) -> IsingModel:
        """Add only external upper-state coupling to the local bias.

        @type subproblem: Subproblem
        @param subproblem: the subproblem of which to add influence to.
        @rtype: IsingModel
        @return: the influenced subproblem.
        """
        expanded = self.expand_upper()
        expanded[subproblem.original_nodes] = 0
        local_influence = self.original_coupling[subproblem.original_nodes] @ expanded
        return IsingModel(subproblem.model.J, subproblem.model.h + local_influence, subproblem.model.c)

    @staticmethod
    def make_meta_nodes(model: IsingModel, nb_meta_nodes: int) -> np.ndarray:
        """Group low-degree seeds with their strongest available signed couplings.

        @type model: IsingModel
        @param model: the model that will be used to cluster into meta-nodes.
        @type nb_meta_nodes: int
        @param nb_meta_nodes: the amount of meta-nodes that need to be made.
        @rtype: np.ndarray
        @return: the partitioning array.
        """
        if not 1 <= nb_meta_nodes <= model.num_variables:
            raise ValueError("Meta-node count must be between one and the number of local nodes")
        coupling = triu_to_symm(model.J)
        sorted_index = np.argsort(np.count_nonzero(coupling, axis=1), kind="stable")
        assigned = np.zeros(model.num_variables, dtype=bool)
        partitions = np.empty(model.num_variables, dtype=int)
        nodes_per_meta, remainder = divmod(model.num_variables, nb_meta_nodes)
        part_id = 0
        for index in sorted_index:
            if assigned[index]:
                continue
            group_size = nodes_per_meta + int(part_id < remainder)
            assigned[index] = True
            partitions[index] = part_id
            neighbors = np.argsort(coupling[index])[::-1]
            selected = neighbors[~assigned[neighbors]][: group_size - 1]
            assigned[selected] = True
            partitions[selected] = part_id
            part_id += 1
        return partitions
