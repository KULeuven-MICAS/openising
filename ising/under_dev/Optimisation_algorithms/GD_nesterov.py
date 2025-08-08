import numpy as np
import tqdm # type: ignore


def GD_nesterov(num_iterations: int, initial_state:np.ndarray, J:np.ndarray, h:np.ndarray, learning_rate: float, momentum:float):
    
    state = np.copy(initial_state)
    speed = np.zeros_like(state)
    for i in tqdm.tqdm(range(num_iterations), leave=True):
        state_ = state + momentum * speed
        grad = -J @ (state_) - h + 3.55*(state_*(state_-1)*(state_+1))

        speed = speed * momentum - learning_rate * grad
        state += speed

        # Apply inelastic walls at extremities
        speed = np.where(np.abs(state) >= 1, 0, speed)
        state = np.where(np.abs(state) >= 1, np.sign(state), state)

    return energy(np.sign(state), J, h), state

def energy(state: np.ndarray, J:np.ndarray, h:np.ndarray) -> float:
    """
    Compute the energy of a given state in the Ising model.
    
    Parameters:
    - state: np.ndarray, the current state of the system (spin configuration).
    - J: np.ndarray, the interaction matrix.
    - h: np.ndarray, the external magnetic field vector.
    
    Returns:
    - energy: float, the energy of the system in the given state.
    """
    return -0.5 * np.dot(state, np.dot(J, state)) - np.dot(h, state)