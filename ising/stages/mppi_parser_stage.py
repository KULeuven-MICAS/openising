from typing import Any
import pathlib
import yaml

from argparse import Namespace

from ising.stages import LOGGER, TOP
from ising.stages.stage import Stage, StageCallable
from ising.stages.model.ising import IsingModel

from ising.stages.model.MPPI.QUBOController import QUBOController
from ising.benchmarks.MPPI import get_dynamics_model
from ising.stages.model.MPPI.environment import generate_random_scene, create_environment, create_reference_trajectory

import numpy as np

class MPPIParserStage(Stage):
    """Parse the MPPI workload stage.

    This functions as the MPPI controller. The stages works as follows

    ----------------------
    |Config Parsing Stage| --> Read configuration.
    ----------------------
    --------------------
    |MPPI Parsing Stage| --> Run MPPI loop w. substages based on configured benchmark.
    --------------------

        -------------------
        |Energy Calc Stage| --> Additional functions for energy calculation
        -------------------

        --------------------
        |Quantization Stage| --> Quantize model based on config
        --------------------

        ------------------
        |Simulation Stage| --> Run solver
        ------------------

    """
    def __init__(self, list_of_callables: list[StageCallable], *, config: Any, **kwargs: Any) -> None:
        super().__init__(list_of_callables, **kwargs)
        self.config = config
        self.benchmark_path = TOP / config.benchmark # Make this a directory
        self.benchmark_filename = str(self.benchmark_path).split('/')[-1]
        # Load benchmark config
        with pathlib.Path(self.benchmark_path).open(encoding="utf-8") as file:
            benchmark: dict = yaml.safe_load(file)
        # Set namespace for benchmark config
        self.benchmark = Namespace(**benchmark)
        self.state_dim = len(self.benchmark.Q) if self.benchmark.Q else len(self.benchmark.Q_diag)
        self.action_dim = len(self.benchmark.R) if self.benchmark.R else len(self.benchmark.R_diag)

    def parse_benchmark_trajectory(self):
        """! Parse and construct the trajectory from config parameters.

        @return: scene object and reference trajectory array
        """
        # Generate scene with fixed amount fo control points and from seed
        scene = generate_random_scene(nb_control_points=self.benchmark.nb_control_points, seed=self.benchmark.seed)
        # Create environment from scene
        env, control_pts, bc_headings = create_environment(scene)
        # Create reference trajectory that navigates the generated scene with configured speed and timestep
        x_ref = create_reference_trajectory(env, control_pts, bc_headings,
                                            v=self.benchmark.velocity, dt=self.benchmark.delta_t)
        return scene, x_ref.T

    def run(self) -> Any:
        """! Parse the benchmark workload and run control loop. """
        LOGGER.debug(f"Parsing MPPI benchmark: {self.benchmark_path}")

        # Get dynamics model from benchmark (defaults to bicycle model)
        model = get_dynamics_model(self.benchmark)

        # Get QUBO parameters from benchmark
        qubo = QUBOController(self.benchmark)
        # Unpack QUBO controller parameters
        rr ,qq, ee = qubo.RR, qubo.QQ, qubo.EE

        # No dummy creation for MPPI
        dummy_creator = self.config.dummy_creator if hasattr(self.config, "dummy_creator") else False
        if dummy_creator:
            raise ValueError("No dummy creation for MPPI.")

        LOGGER.debug(f"Parsing MPPI trajectory: {self.benchmark_filename}")
        # Reference benchmark trajectory
        scene, x_ref = self.parse_benchmark_trajectory()
        # Takes first point of reference as starting point for control loop (Could also be made configurable)
        executed_trajectory = [x_ref[0, :]]
        # Initialize a list to keep every rollout for visualization
        predicted_trajectory = []
        # Initialize action sequence (defaults to None which results in random sampling)
        u_bar = None

        LOGGER.debug(f"Running trajectory with length {x_ref.shape[0]}")
        # Iterate over reference points
        for point in np.arange(start=1, stop=x_ref.shape[0], step=self.benchmark.action_horizon):
            # Most recently visited state
            state = executed_trajectory[-1]
            # Get horizon view
            x_ref_view = self.get_trajectory_view(x_ref[point:, :])
            # Set initial actions to zero
            u_bar = self.reset_actions(u_bar)
            # Amount of MPPI iterations is defined in benchmark config
            for iteration in range(self.benchmark.n_mppi_iterations):
                # Empty variation holder for vjp
                dx, du = state[None, ...], u_bar[None, ...]
                # Do model rollout
                x_bar, fxu_bar, A_seq, B_seq, _ = model.rollout(state, u_bar, dx, du)
                # Build ising model
                ising_model = self.build_ising(
                    x_bar, x_ref_view, A_seq, B_seq, rr, qq, ee
                )
                # Necessary kwargs for next stages
                self.kwargs["config"] = self.config
                self.kwargs["ising_model"] = ising_model
                # Get sub stages --> See workflow
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
                # Run solver sub stages
                ans, debug_info = next(sub_stage.run())
                # Get minimal energy for first used solver (Could iterate over all solvers, but is unnecessary)
                solver = self.config.solvers[0]
                # Take minimal energy
                best_idx = np.argmin(ans.energies[solver])
                # Cast actions from {-1, +1} to {0, +1}
                actions = (ans.states[solver][best_idx] + 1.0) / 2.0
                # Apply actions in continuous space
                u_bar = u_bar + (actions @ ee.T).reshape(-1, self.action_dim)

            # Execute actions
            for a in range(self.benchmark.action_horizon):
                # Select actions
                new_u = u_bar[a, :]
                # Apply forward model
                state, force = model.discrete_step(state.squeeze(), new_u.squeeze())
                # Add new state to list
                executed_trajectory.append(state)

            # Add full predicted trajectory at current point
            predicted_trajectory.append(
                x_bar.reshape(-1, self.state_dim)
            )
        # Add final result to answer (and some stuff for result plotting)
        ans.executed_trajectory = executed_trajectory
        ans.predicted_trajectory = predicted_trajectory
        ans.reference_trajectory = x_ref
        ans.scene = scene
        ans.delta_t = self.benchmark.delta_t
        yield ans, debug_info

    def get_trajectory_view(self, trajectory: np.ndarray) -> np.ndarray:
        """! Return view of trajectory.
         This gets to next part of the reference within horizon length.
         The view is tiled if the remaining points are less than horizon length.

        @type trajectory: np.ndarray
        @param trajectory: remaining trajectory (T, state_dim)
        @rtype: np.ndarray
        @return: View of the trajectory with horizon length (L, state_dim)
        """
        # Get horizon
        L = self.benchmark.HORIZON_LENGTH
        # Get horizon view
        view = trajectory[:L, :]
        # If selected reference view is smaller than horizon
        if view.shape[1] < L:
            last_col = view[-1:, :]
            num_pad = L - view.shape[0]
            view = np.vstack((view, np.repeat(last_col, num_pad, axis=0)))
        # Flatten and put T first
        return view.reshape(-1, 1)

    def reset_actions(self, prev_actions: np.ndarray|None = None) -> np.ndarray:
        """! Reset all actions. Re-uses actions when given.

        @type prev_actions: np.ndarray|None
        @param prev_actions: previous action sequence (defaults to None)
        @rtype: np.ndarray
        @return: New actions sequence which is either fully zero or only the new actions at end of the horizon.
        """
        # Horizon length
        L = self.benchmark.HORIZON_LENGTH
        # Zero actions
        actions = np.zeros((L, self.action_dim))
        # Set non executed actions as initial state for next iteration
        if prev_actions is not None:
            actions[:-self.benchmark.action_horizon, :] = prev_actions[self.benchmark.action_horizon:, :]
        return actions

    def build_ising(self,
                    x_bar: np.ndarray,
                    x_ref: np.ndarray,
                    A_seq: np.ndarray,
                    B_seq: np.ndarray,
                    RR: np.ndarray,
                    QQ: np.ndarray,
                    EE: np.ndarray
                    ) -> IsingModel:
        """! Build Ising model.

        @type state: np.ndarray
        @param current state: Current state of trajectory (state_dim,)
        @type x_bar: np.ndarray
        @param x_bar: Nominal trajectory rollout (L + 1, state_dim)
        @type x_ref: np.ndarray
        @param x_ref: View of the reference trajectory (L, state_dim)
        @type A_seq: np.ndarray
        @param A_seq: Sequence of A matrices (dz/dz) (L, state_dim, state_dim)
        @type B_seq: np.ndarray
        @param B_seq: Sequence of B matrices (dz/da) (L, state_dim, action_dim)
        @type RR: np.ndarray
        @param RR: Cost matrix for actions (action_dim, action_dim)
        @type QQ: np.ndarray
        @param QQ: Cost matrix for states (state_dim, state_dim)
        @type EE: np.ndarray
        @param EE: Binary encoding matric (state_dim * n_bits, n_bits)

        @rtype: IsingModel
        @return: Ising Model to run
        """
        # Check if approximation is done
        assert self.benchmark.n_approx_iter is not None
        # Build forward state transitions
        phi_fwd, phi_bwd = build_phi(A_seq, self.benchmark.delta_t, n_approx_iter=self.benchmark.n_approx_iter)
        # Make into arrays
        phi_fwd_arr, phi_bwd_arr = np.stack(phi_fwd, axis=0), np.stack(phi_bwd, axis=0)
        # Build B from forward state transitions and action Jacobians
        B_flat, _ = build_B_mat(phi_fwd_arr, phi_bwd_arr, B_seq, self.benchmark.delta_t)
        # Build QUBO problem for reference tracking
        J, h, c = self.build_qubo(
            x_bar,
            x_ref,
            B_flat,
            QQ, RR,
            E=EE,
        )
        # Absorb diagonal
        Q = J + np.diag(h)
        # Build ising mdoel from QUBO in framework
        return IsingModel.from_qubo(Q)

    def build_qubo(
            self,
            x_bar: np.ndarray,
            x_ref: np.ndarray,
            B_flat: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            E:np.ndarray | None = None,
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @type x_bar: np.ndarray
        @param x_bar: Nominal trajectory from rollout (L + 1, state_dim)
        @type x_ref: np.ndarray
        @param x_ref: View of the reference trajectory (L + 1, state_dim)
        @type B_flat: np.ndarray
        @param B_flat: Full interaction matrix (L, state_dim, action_dim)
        @type Q: np.ndarray
        @param Q: Cost matrix for states (state_dim, state_dim)
        @type R: np.ndarray
        @param R: Cost matrix for actions (action_dim, action_dim)
        @type E: np.ndarray
        @param E: Optional binary encoding matrix (state_dim * n_bits, n_bits)

        @rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        @return: tuple with quadratic, linear and constant part of the ising model
        """
        # Make residual. This drops current point (x_state = x_bar[0] be design)
        r = x_bar[1:].reshape(-1) - x_ref.reshape(-1)
        # Apply binary encoding if given
        BE = B_flat if E is None else B_flat @ E  # (T*state_dim, T*action_dim) or (T*state_dim, n_bits)

        # Build the QUBO terms (See paper for maths)
        QBE = Q @ BE  # (T*state_dim, n_cols)
        J_mat = BE.T @ QBE  # (n_cols, n_cols)  quadratic
        J_mat = J_mat + (E.T @ R @ E if E is not None else R)  # + R̃ or E^T R̃ E
        h = 2.0 * (BE.T @ (Q @ r))  # (n_cols,)         linear
        c = r @ (Q @ r)  # constant

        # Make symmetric
        J_mat = J_mat.T + J_mat
        # Return quadratic, linear and constant parts
        return J_mat, h, c



######################################
# Functions for building MPPI solution
######################################

def build_phi(A_seq: np.ndarray, dt: float, n_approx_iter: int = 1)->tuple[np.ndarray, np.ndarray]:
    """! Build forward and backward rollout sequences.

    Notes
    -----
    Φ_k = Pi_j=0^k I + A_j * dt
    phi_fwd[k] = Φ_k @ Φ_{k-1} @ ... @ Φ_0     (left-product, x_{k+1} = phi_fwd[k] @ x_0)
    phi_bwd[k] ≈ phi_fwd[k]^{-1}                (approximate inverse)

    First-order approx (n_approx_iter=1):  Φ_k^{-1} ≈ I - A_k·dt   (error O(dt²))
    Higher-order (n_approx_iter=m):        adds terms up to (-A·dt)^m (error O(dt^{m+1}))

    @type A_seq: np.ndarray
    @param A_seq: sequence of state transition Jacobians (L, state_dim, state_dim)
    @type dt: float
    @param dt: Time step
    @type n_approx_iter: int
    @param n_approx_iter: Number of approximate iterations for inverse

    @rtype: tuple[np.ndarray, np.ndarray]
    @return: Forward and backward transitions as defined in notes.
    """
    # Length and dimension
    T, nx, _ = A_seq.shape
    # Make unit matrix
    I_mat = np.eye(nx)

    # Function to scan over forward transitions
    def scan_fwd(phi_prev: np.ndarray, A_k: np.ndarray) -> np.ndarray:
        """! Scan function for forward state transitions.

        @type phi_prev: np.ndarray
        @param phi_prev: previous full state transition (state_dim, state_dim)
        @type A_k: np.ndarray
        @param A_k: current state transition Jacobian (state_dim, state_dim)

        @rtype: np.ndarray
        @return: Forward state transition (as defined in notes).
        """
        phi_next = (I_mat + A_k * dt) @ phi_prev  # left-multiply: newer step on left
        return phi_next

    # Start with unit
    phi_prev = I_mat
    # Empty forward transitions
    phi_fwd = []
    # Scan over sequence transitions
    for A in A_seq:
        # Use scan function and append
        phi_next = scan_fwd(phi_prev, A)
        phi_fwd.append(phi_next)
        # Set current transition
        phi_prev = phi_next

    # Calculate inverse
    def inv_factor(A_k: np.ndarray) -> np.ndarray:
        """! Calculate inverse transition matrix to order n_approx_iter.

        @type A_k: np.ndarray
        @param A_k: Forward state transition Jacobian

        @rtype: np.ndarray
        @return Inverse state transition matrix (I + A_k·dt)^{-1} to order n_approx_iter in dt."""
        # Geometric series: (I + X)^{-1} = I - X + X² - ...  where X = A_k·dt
        X = A_k * dt
        result = I_mat - X
        if n_approx_iter > 1:
            Xpow = X
            for _ in range(n_approx_iter - 1):
                Xpow = Xpow @ X
                # alternating sign
                result = result + Xpow if (_ % 2 == 0) else result - Xpow
        return result

    # Scan function for backwards transitions
    def scan_bwd(phi_bwd_prev: np.ndarray, A_k:np.ndarray) -> np.ndarray:
        """! Scan function for backward state transitions.

        @type phi_prev: np.ndarray
        @param phi_prev: previous full state transition in backwards order (state_dim, state_dim)
        @type A_k: np.ndarray
        @param A_k: current state transition Jacobian (state_dim, state_dim)

        @rtype: np.ndarray
        @return: backwards propagated state transition (as defined in notes).
        """
        phi_bwd_next = phi_bwd_prev @ inv_factor(A_k)
        return phi_bwd_next

    # Start with identity
    phi_prev = I_mat
    # Empty sequence
    phi_bwd = []
    # Iterate over Jacobians
    for A in A_seq:
        phi_next = scan_bwd(phi_prev, A)
        phi_bwd.append(phi_next)
        phi_prev = phi_next
    # Return backwards and forward sequence
    return phi_fwd, phi_bwd


def build_B_mat(phi_fwd, phi_bwd, B_seq, dt):
    """! Build causal linearized control influence matrix B_mat.
    B_mat[i,j] = phi_fwd[i] @ phi_bwd[j] @ B_seq[j] * dt

    @type phi_fwd: np.ndarray
    @param phi_fwd: Forward propagated state transitions (L, nx, nx)
    @type phi_bwd: np.ndarray
    @param phi_bwd: Backward propagated state transitions (L, nx, nx)
    @type B_seq: np.ndarray
    @param B_seq: Control Jacobians (dz/da) (L, nx, nu)
    @type dt: float
    @param dt: Time step

    @rtype: np.ndarray
    @return: Causal interactions matrix for system (L, L, nx, nu), reshaped to (L*state_dim, L*action_dim).
    """
    # This is the "B column for each time" pre-multiplied by the backward state transitions
    phi_bwd_B = np.einsum('jmk,jku->jmu', phi_bwd, B_seq) * dt

    # B[i,j] = phi_fwd[i] @ phi_bwd_B[j] (outer product)
    B = np.einsum('inm,jmu->ijnu', phi_fwd, phi_bwd_B)

    # Causal mask
    T = phi_fwd.shape[0]
    causal_mask = np.tril(np.ones((T, T), dtype=phi_fwd.dtype))
    B = B * causal_mask[:, :, None, None]

    # Reshape to matrix
    B_flat = B.transpose(0, 2, 1, 3).reshape(B.shape[0] * B.shape[2],
                                             B.shape[1] * B.shape[3])
    return B_flat, B  # return both for debugging
