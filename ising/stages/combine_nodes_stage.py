from ising.stages import LOGGER
from typing import Any
import numpy as np
import math
import copy
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel


class CombineNodesStage(Stage):
    """! Stage to quantize the Ising model."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        config: Any,
        ising_model: IsingModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.ising_model = ising_model

    def run(self) -> Any:
        """! Double the amount of nodes and redistribute the connections."""

        if self.config.combine_nodes:
            nodes_scaling = self.config.nodes_scaling if hasattr(self.config, "nodes_scaling") else 2
            self.config.nodes_scaling = nodes_scaling
            original_J = self.ising_model.J
            original_h = self.ising_model.h
            new_J, new_h = self.split_nodes(original_J, original_h, nodes_scaling)

            split_model = IsingModel(
                J=np.triu(new_J, k=1),
                h=new_h,
                c=self.ising_model.c,
            )
        else:
            LOGGER.debug("Split Nodes is disabled.")
            split_model = copy.deepcopy(self.ising_model)

        self.kwargs["config"] = self.config
        self.kwargs["ising_model"] = split_model
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for ans, debug_info in sub_stage.run():
            ans.ising_model = self.ising_model
            ans.split_model = split_model
            if self.config.combine_nodes:
                for solver in ans.config.solvers:
                    for energy_id in range(len(ans.energies[solver])):
                        orig_state = self.translate_state(ans.states[solver][energy_id], nodes_scaling=nodes_scaling)
                        ans.energies[solver][energy_id] = self.ising_model.evaluate(orig_state.astype(np.float32))
                        if hasattr(ans, "tsp_energies"):
                            if ans.tsp_energies[solver][energy_id] == math.inf:
                                ans.tsp_energies[solver][energy_id] = math.inf
                            else:
                                ans.tsp_energies[solver][energy_id] = ans.energies[solver][energy_id]
                        ans.states[solver][energy_id] = orig_state
            yield ans, debug_info

    def split_nodes(self, J: np.ndarray, h: np.ndarray, nodes_scaling: int) -> tuple[np.ndarray, np.ndarray]:
        """! Split nodes in the Ising model.

        @type J: np.ndarray
        @param J: Coupling matrix of the Ising model.
        @type h: np.ndarray
        @param h: Local field vector of the Ising model.
        @type nodes_scaling: int
        @param nodes_scaling: number of nodes to split nodes into.
        @rtype: tuple[np.ndarray, np.ndarray]
        @return: new coupling matrix and local field vector.
        """
        nb_nodes = J.shape[0]
        new_nb_nodes = nodes_scaling * nb_nodes
        new_J = np.zeros((new_nb_nodes, new_nb_nodes))
        new_h = np.zeros(new_nb_nodes)

        for i in range(nb_nodes):
            for j in range(i + 1, nb_nodes):
                # Distribute J[i,j] across an (nodes_scaling x nodes_scaling) block so
                # that the sum of the block equals the original value.
                # We assume J contains integers. Distribute the integer value
                # across the (nodes_scaling x nodes_scaling) block so the block
                # entries are integers and sum exactly to the original value.
                # Build a symmetric integer block whose entries sum to J[i,j].
                # Prefer putting the smallest values on the diagonal by
                # distributing extra units to off-diagonal symmetric pairs first.
                sign = np.sign(J[i, j])
                abs_total = np.abs(J[i, j])

                base = abs_total // nodes_scaling**2
                remainder = abs_total - base * nodes_scaling**2

                # Start with base everywhere
                block = np.full((nodes_scaling, nodes_scaling), base, dtype=int)

                # Distribute +1 to symmetric off-diagonal pairs first (consume 2 from remainder)
                for m in range(nodes_scaling):
                    for n in range(m + 1, nodes_scaling):
                        if remainder >= 2:
                            block[m, n] += 1
                            block[n, m] += 1
                            remainder -= 2
                        else:
                            break

                # If one unit remains, put it on a diagonal element
                if remainder == 1:
                    # place on first diagonal (deterministic)
                    block[0, 0] += 1

                # Assign symmetric blocks in the big matrix
                new_J[
                    nodes_scaling * i : nodes_scaling * i + nodes_scaling,
                    nodes_scaling * j : nodes_scaling * j + nodes_scaling,
                ] = sign * block
                new_J[
                    nodes_scaling * j : nodes_scaling * j + nodes_scaling,
                    nodes_scaling * i : nodes_scaling * i + nodes_scaling,
                ] = sign * block.T

            # divide h[i] over the new ndoes
            sign_h = np.sign(h[i])
            abs_total_h = np.abs(h[i])
            base_h = abs_total_h // nodes_scaling
            remainder_h = abs_total_h % nodes_scaling
            new_h[nodes_scaling * i : nodes_scaling * i + nodes_scaling] = (
                np.ones((nodes_scaling,)) * base_h
            )
            for m in range(nodes_scaling):
                if remainder_h > 0:
                    new_h[nodes_scaling*i + m] += 1
                    remainder_h -= 1
                else:
                    break
            new_h[nodes_scaling * i : nodes_scaling * i + nodes_scaling] *= sign_h

        # Set the diagonal elements to the maximum element available
        max_J = np.max(np.abs(new_J))
        diag_part = np.triu(np.ones((nodes_scaling, nodes_scaling)) * (max_J), 1)
        for i in range(nb_nodes):
            new_J[
                nodes_scaling * i : nodes_scaling * i + nodes_scaling,
                nodes_scaling * i : nodes_scaling * i + nodes_scaling,
            ] = diag_part

        return new_J, new_h

    def translate_state(self, state_split: np.ndarray, nodes_scaling: int) -> np.ndarray:
        """Translates the states from the split version to the original version. Each spin takes
        the average value of all its replica spins (majority vote).

        @type state_split: np.ndarray
        @param state_split: the state of the split model.
        @type nodes_scaling: int
        @param nodes_scaling: the amount of splits per original node.
        @rtype: np.ndarray
        @return: the state of the original model.
        """
        result = np.zeros((int(len(state_split) / nodes_scaling),), dtype=np.float32)
        if nodes_scaling == 2:
            for i in range(len(result)):
                result[i] = state_split[nodes_scaling * i]
        else:
            for i in range(len(result)):
                result[i] = np.bincount(state_split[nodes_scaling * i : nodes_scaling * i + nodes_scaling].argmax())
        return result
