import jax
import jax.numpy as jnp

from functools import partial


@jax.tree_util.register_pytree_node_class
class DynamicsModel:
    def __init__(self, dt):
        self.dt = dt

    # --- PyTree Requirements ---
    def tree_flatten(self):
        # Defines what is data (leaves) and what is metadata (aux_data)
        # Leaves are things JAX can differentiate or transform.
        children = (self.dt,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Tells JAX how to rebuild the object
        return cls(*children)

    @jax.jit
    def force(self, state, action):
        raise NotImplementedError("Force calculation needs to be implemented.")

    @jax.jit
    def jacobian(self, state, action):
        """
        Simple holder for jacobian calculation.
        Should try to use jvp if possible.
        Returns A, B at (s, a) -> Should be vmap-able
        """
        args = [state, action]
        return jax.jacfwd(self.force, argnums=(0, 1))(*args)

    @jax.jit
    def discrete_step(self, state, action):
        """
        Simple euler dynamics for forward stepping.
        Stores jvp functions around linearization point.
        vjp_fn(dx, du) = Ads + Bda at (s, a)
        """
        # Discrete time step
        force = self.force(state, action)
        # Return updated state and linearized function at state, action point
        return state + force * self.dt, force

    @partial(jax.jit, static_argnums=(3,))
    def _rollout(self, init_state, actions, return_jacobian: bool = True):
        """
        Scans over actions, returning the trajectory and the explicit
        return_jacobians: bool Flag to return discrete Jacobians A and B along the path.
        """

        def scan_fn(current_state, action):
            # Forward dynamics
            next_state, force = self.discrete_step(current_state, action)

            if return_jacobian:
                # Calculate explicit continuous Jacobians
                A, B = self.jacobian(current_state, action)
                return next_state, (current_state, force, A, B)

            # Only return carry and current state
            return next_state, (current_state, force)

        # Run the scan
        final_state, output = jax.lax.scan(scan_fn, init_state, actions)
        trajectory, forces = output[0:2]
        # Append the final state to match the [N+1, n_x] shape
        full_trajectory = jnp.vstack([trajectory, final_state])
        if return_jacobian:
            A_seq, B_seq = output[2], output[3]
            return (full_trajectory, forces), (A_seq, B_seq)
        return full_trajectory, forces

    @jax.jit
    def rollout(self, init_state, actions, dx, du, return_jacobian=True):
        """
        Linearized rollout that returns f(x), vjp_fn(dx, du).
        vjp_fn(dx, du) = Adx + Bdu
        """
        # Standard *return_jacobians* is True
        (full_trajectory, forces), vjp_fn, (A_seq, B_seq) = jax.linearize(self._rollout, init_state, actions,
                                                                          has_aux=return_jacobian)
        # Return
        return full_trajectory, forces, A_seq, B_seq, jax.vmap(vjp_fn, in_axes=(0, 0))(dx, du)


@jax.tree_util.register_pytree_node_class
class BicycleModel(DynamicsModel):
    def __init__(self, dt):
        super().__init__(dt)

    def force(self, state, action):
        """Bicycle steering forward dynamics model. """
        # state = [x position, y position, heading, speed, steering angle]
        # action = [acceleration, steering velocity]
        _, _, heading, velocity, angle = state
        return jnp.array([
            velocity * jnp.cos(heading),
            velocity * jnp.sin(heading),
            velocity * jnp.tan(angle),
            action[0],
            action[1]
        ])


@jax.tree_util.register_pytree_node_class
class CartesianAcceleratorModel(DynamicsModel):
    def __init__(self, n_dim, dt):
        super().__init__(dt)
        self.n_dim = n_dim

    def force(self, state, action):
        """Bicycle steering forward dynamics model. """
        # state = [*position, *velocity]
        # action = [*acceleration]
        # position = state[:self.n_dim]
        velocity = state[self.n_dim:]
        return jnp.concatenate([velocity, action])
