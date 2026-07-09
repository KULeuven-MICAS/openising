import numpy as np
from typing import Any

class QUBOController:
    """QUBO Controller instance"""
    def __init__(self, config: Any):
        """
        Initialise the controller.

        param controller_config: controller config class
        param qubo_config: qubo config class
        """
        self.config = config

        self.EE, self.mapping = self.continuous_to_binary_mapping(
            config.action_dim,
            config.HORIZON_LENGTH,
            config.K,
            config.N_BITS,
            config.action_scales,
            config.scale_multiplier
        )

        self.Q_slack = config.Q_slack

        self._RR = self.build_R_trajectory(None, None)
        self._QQ = self.build_Q_trajectory(None, None)

    @property
    def RR(self):
        return self._RR

    @property
    def QQ(self):
        return self.Q_slack * self._QQ

    # Clipping functions
    def clip_actions(self, actions: np.ndarray) -> np.ndarray:
        return self._clip_bounds(actions,
                           self.config.Q_BOUNDS,
                           self.config.variable_to_action_map)

    def clip_states(self, states: np.ndarray) -> np.ndarray:
        return self._clip_bounds(states,
                           self.config.R_BOUNDS,
                           self.config.variable_to_state_map)

    def build_Q_trajectory(self, x_bar, u_bar):
        if self.config.Q is None:
            if self.config.Q_diag is not None:
                Q = self.Q_slack * np.diag(self.config.Q_diag)
            else:
                Q = self.Q_slack * np.eye(self.config.state_dim)
        else:
            Q = self.config.Q

        # State weights I_N ⊗ Q w. last diagonal term weighted (HORIZON_LENGTH * state_dim, HORIZON_LENGTH * state_dim)
        W = np.eye(self.config.HORIZON_LENGTH)
        W[-1, -1] = self.config.TERMINAL_WEIGHT
        QQ = np.kron(W, Q)
        return QQ

    def build_R_trajectory(self, x_bar, u_bar):
        if self.config.R is None:
            if self.config.R_diag is not None:
                R = np.diag(self.config.R_diag)
            else:
                R = np.eye(self.config.action_dim)
        else:
            R = self.config.R

        # Action weights: I_N ⊗ R (HORIZON_LENGTH * action_dim, HORIZON_LENGTH * action_dim)
        RR = np.kron(np.eye(self.config.HORIZON_LENGTH), R)
        return RR
    def continuous_to_binary_mapping(
            self,
            m: int,
            HORIZON_LENGTH: int,
            BIT_MULTIPLIER: int,
            N_BITS: int,
            action_scales: list[float] | None = None,
            scale_multiplier: float|None = None,
    ):
        # Setup default scales
        if action_scales is None:
            action_scales = [20.0, 2.0] if m == 2 else [1.0] * m
        if scale_multiplier is None:
            scale_multiplier = 1./3.

        scales = np.array(action_scales) * np.array(scale_multiplier)  # Shape (m,)

        # Bit vector
        bits = np.arange(N_BITS)
        E_base = BIT_MULTIPLIER * (2.0 ** bits)
        E_base = E_base / E_base.max() / 2.0
        # MSB flip for 2's complement style mapping
        E_base[..., -1] = -E_base[..., -1]

        # Spatial block E (m x m*L)
        # kron(diag(scales), E_base) -> (m, m) ⊗ (1, N_BITS) = (m, m*N_BITS)
        E_block = np.kron(np.diag(scales), E_base[None, :])

        # Block diagonal EE (m*N x m*L*N)
        # kron(I_N, E_block) -> (HORIZON_LENGTH, HORIZON_LENGTH) ⊗ (m, m*N_BITS) = (m*N, m*N_BITS*HORIZON_LENGTH)
        EE = np.kron(np.eye(HORIZON_LENGTH), E_block)
        return EE, {}

    def _clip_bounds(self, arr, bounds, variable_map):
        for variable, bounds in bounds.items():
            # Variable mapping: added complexity for flexibility
            i = variable_map[variable]
            # Clip to bounds
            np.clip(arr[:, i], bounds[0], bounds[1], out=arr[:, i])
        return arr

