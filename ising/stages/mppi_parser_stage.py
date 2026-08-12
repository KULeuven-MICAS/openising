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
        # Set namespace and
        self.benchmark = Namespace(**benchmark)

    def parse_benchmark_trajectory(self):
        scene = generate_random_scene(nb_control_points=self.benchmark.nb_control_points, seed=self.benchmark.seed)
        env, control_pts, bc_headings = create_environment(scene)
        x_ref = create_reference_trajectory(env, control_pts, bc_headings,
                                            v=self.benchmark.velocity, dt=self.benchmark.delta_t)
        return scene, x_ref.T

    def run(self) -> Any:
        """Parse the benchmark workload."""
        LOGGER.debug(f"Parsing MPPI benchmark: {self.benchmark_path}")

        # Get dynamics model
        model = get_dynamics_model(self.benchmark)

        # Get QUBO parameters
        qubo = QUBOController(self.benchmark)
        # Unpack QUBO controller parameters
        rr ,qq, ee = qubo.RR, qubo.QQ, qubo.EE

        # No dummy creation for MPPI
        dummy_creator = self.config.dummy_creator if hasattr(self.config, "dummy_creator") else False
        if dummy_creator:
            raise ValueError("No dummy creation for MPPI.")

        LOGGER.debug(f"Parsing MPPI trajectory: {self.benchmark_filename}")
        scene, x_ref = self.parse_benchmark_trajectory() # Reference benchmark trajectory
        executed_trajectory = [x_ref[0, :]] # Initial state (Can be benchmark.x_init)
        predicted_trajectory = [] # List of all rollouts
        u_bar = None # Initial actions (Can be benchmark.u_init)

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
                # Store answers
                ans, debug_info = next(sub_stage.run()) # This runs the ising model
                # Run solver sub stages
                # TODO: implement strategy for multiple solvers
                # for solver in solver:
                #     energies = ans.energies[solver]
                solver = self.config.solvers[0]
                best_idx = np.argmin(ans.energies[solver])
                actions = (ans.states[solver][best_idx] + 1.0) / 2.0
                # Apply actions in continuous space
                u_bar = u_bar + (actions @ ee.T).reshape(-1, self.benchmark.action_dim)

            # Execute actions
            for a in range(self.benchmark.action_horizon):
                new_u = u_bar[a, :]
                state, force = model.discrete_step(state.squeeze(), new_u.squeeze())
                # Add new state to list
                executed_trajectory.append(state)

            # Full predicted trajectory at point
            predicted_trajectory.append(
                x_bar.reshape(-1, self.benchmark.state_dim)
            )
        # Add final result to answer (and some stuff for result plotting)
        ans.executed_trajectory = executed_trajectory
        ans.predicted_trajectory = predicted_trajectory
        ans.reference_trajectory = x_ref
        ans.scene = scene
        ans.delta_t = self.benchmark.delta_t
        yield ans, debug_info

    def get_trajectory_view(self, trajectory: np.ndarray) -> np.ndarray:
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
        L = self.benchmark.HORIZON_LENGTH
        action_dim = self.benchmark.action_dim
        actions = np.zeros((L, action_dim))
        # Set non selected actions as initial state for next iteration
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
        """Build Ising model.

        Args:
            state: current state (state_dim,)
            x_bar: Nominal trajectory rollout (L + 1, state_dim)
            x_ref: Windowed reference trajectory (L, state_dim)
            A_seq: Sequence of A matrices (L, state_dim, state_dim)
            B_seq: Sequence of B matrices (L, state_dim, action_dim)
            RR: Cost matrix for actions (action_dim, action_dim)
            QQ: Cost matrix for states (state_dim, state_dim)
            EE: Binary encoding matric (state_dim * n_bits, n_bits)

        Returns:
            Ising Model to run
        """
        # Check if approximation is done
        assert self.benchmark.n_approx_iter is not None
        # Build phi
        phi_fwd, phi_bwd = build_phi(A_seq, self.benchmark.delta_t, n_approx_iter=self.benchmark.n_approx_iter)
        phi_fwd_arr, phi_bwd_arr = np.stack(phi_fwd, axis=0), np.stack(phi_bwd, axis=0)
        # Build B
        B_flat, _ = build_B_mat(phi_fwd_arr, phi_bwd_arr, B_seq, self.benchmark.delta_t)
        # Build J and h
        J, h, c = self.build_qubo(
            x_bar,
            x_ref,
            B_flat,
            QQ, RR,
            E=EE,
        )
        Q = J + np.diag(h)
        return IsingModel.from_qubo(Q)

    def build_qubo(
            self,
        x_bar,        # (T+1, nx)   nominal trajectory from rollout
        x_ref,        # (T, nx)     reference trajectory (horizon window)
        B_flat,       # (T*nx, T*nu) or (T*nx, n_bits) if pre-multiplied by E
        Q, R,         # (T*nx, T*nx), (T*nu, T*nu) cost matrices
        E=None,       # (T*nu, n_bits) encoding matrix, None for continuous
    ):
        # --- Residual terms ---
        r = x_bar[1:].reshape(-1) - x_ref.reshape(-1)  # (T*state_dim,)
        # --- binary encoding ---
        BE = B_flat if E is None else B_flat @ E  # (T*state_dim, T*action_dim) or (T*state_dim, n_bits)

        # --- QUBO terms ---
        QBE = Q @ BE  # (T*state_dim, n_cols)
        J_mat = BE.T @ QBE  # (n_cols, n_cols)  quadratic
        J_mat = J_mat + (E.T @ R @ E if E is not None else R)  # + R̃ or E^T R̃ E
        h = 2.0 * (BE.T @ (Q @ r))  # (n_cols,)         linear
        c = r @ (Q @ r)  # scalar            constant

        # --- Symmetrize ---
        J_mat = J_mat.T + J_mat
        return J_mat, h, c



######################################
# Functions for building MPPI solution
######################################

def build_phi(A_seq, dt, n_approx_iter: int = 1):
    """
    Build forward and backward rollout sequences.

    Φ_k = Pi_j=0^k I + A_j * dt
    phi_fwd[k] = Φ_k @ Φ_{k-1} @ ... @ Φ_0     (left-product, x_{k+1} = phi_fwd[k] @ x_0)
    phi_bwd[k] ≈ phi_fwd[k]^{-1}                (approximate inverse)

    First-order approx (n_approx_iter=1):  Φ_k^{-1} ≈ I - A_k·dt   (error O(dt²))
    Higher-order (n_approx_iter=m):        adds terms up to (-A·dt)^m (error O(dt^{m+1}))
    """
    T, nx, _ = A_seq.shape
    I_mat = np.eye(nx)

    def scan_fwd(phi_prev, A_k):
        phi_next = (I_mat + A_k * dt) @ phi_prev  # left-multiply: newer step on left
        return phi_next

    phi_prev = I_mat
    phi_fwd = []
    for A in A_seq:
        phi_next = scan_fwd(phi_prev, A)
        phi_fwd.append(phi_next)
        phi_prev = phi_next

    def inv_factor(A_k):
        """(I + A_k·dt)^{-1} to order n_approx_iter in dt."""
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

    def scan_bwd(phi_bwd_prev, A_k):
        # phi_bwd[k] = phi_bwd[k-1] @ Φ_k^{-1}  (right-multiply: older inverse on right)
        # This gives phi_fwd[k]^{-1} = Φ_0^{-1} @ ... @ Φ_k^{-1}
        phi_bwd_next = phi_bwd_prev @ inv_factor(A_k)
        return phi_bwd_next

    phi_prev = I_mat
    phi_bwd = []
    for A in A_seq:
        phi_next = scan_bwd(phi_prev, A)
        phi_bwd.append(phi_next)
        phi_prev = phi_next

    return phi_fwd, phi_bwd


def build_B_mat(phi_fwd, phi_bwd, B_seq, dt):
    """
    Build linearized control influence matrix B_mat.
    B_mat[i,j,n,u] = phi_fwd[i] @ phi_bwd[j] @ B_seq[j] * dt   for j <= i
                   = 0                                            for j >  i

    Shapes:
        phi_fwd : (T, nx, nx)
        phi_bwd : (T, nx, nx)
        B_seq   : (T, nx, nu)
        output  : (T, T, nx, nu) → reshape to (T*nx, T*nu)
    """
    # Step 1: phi_bwd[j] @ B_seq[j] * dt  for all j  →  (T, nx, nu)
    # This is the "B column for each time" pre-multiplied by the inverse STM
    phi_bwd_B = np.einsum('jmk,jku->jmu', phi_bwd, B_seq) * dt  # (T, nx, nu)

    # Step 2: outer product over (i,j) pairs
    # B[i,j,n,u] = phi_fwd[i,n,m] * phi_bwd_B[j,m,u]
    B = np.einsum('inm,jmu->ijnu', phi_fwd, phi_bwd_B)  # (T, T, nx, nu)

    # Step 3: causal mask — action j can only affect state i if j <= i
    T = phi_fwd.shape[0]
    causal_mask = np.tril(np.ones((T, T), dtype=phi_fwd.dtype))
    B = B * causal_mask[:, :, None, None]

    # Reshape to (T*nx, T*nu): state dim is outer, action dim is inner
    # B[i,j,n,u] → B_flat[(i*nx + n), (j*nu + u)]
    B_flat = B.transpose(0, 2, 1, 3).reshape(B.shape[0] * B.shape[2],
                                             B.shape[1] * B.shape[3])
    return B_flat, B  # return both for debugging
