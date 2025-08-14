import numpy as np
import matplotlib.pyplot as plt
import logging

from ising.flow import LOGGER
from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.solvers.Multiplicative import Multiplicative

logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

def decode(M, xtilde):
    r = int(np.ceil(np.log2(np.sqrt(M))))
    shift = np.sqrt(M) - 1
    x_shifted = xtilde + shift

    # Step 2: Undo scaling from T (which is equivalent to decoding base-2 integers)
    # Since T * (sigma + 1) = sum_{i=1 to r} 2^{r-i} * (sigma_i + 1)
    # Therefore, (x_shifted / 2) gives us the integer encoded by the r bits
    x_int = (x_shifted / 2).astype(int)  # shape: (N,)

    # Step 3: Convert each integer to r-bit binary
    bits = ((x_int[:, None] >> np.arange(r-1, -1, -1)) & 1)  # shape: (N, r)

    # Step 4: Convert to bipolar {-1, 1} and flatten in column-major (Fortran) order
    sigma = (2 * bits - 1).reshape(-1, order='F')  # shape: (r*N,)

    return sigma

def plot_energy(model, optimal_state, received_state, optimal_energy):
    N = model.num_variables
    num_states = 2**N

    # Generate all possible states
    states = []
    energies = []
    optimal_index = -1
    received_index = -1
    received_energy = None
    # Use binary numbers to generate all combinations of -1 and 1
    for i in range(num_states):
        # Convert number to binary and pad with zeros
        binary = format(i, f'0{N}b')
        # Convert 0s and 1s to -1s and 1s
        state = np.array([1 if int(b) else -1 for b in binary])
        energy = model.evaluate(state)
        if np.all(state == optimal_state):
            optimal_index = i
        if np.all(state == received_state):
            received_index = i
            received_energy = energy
        if energy <= optimal_energy and np.any(state != optimal_state):
            LOGGER.info(f"Found state that has optimal energy: {state}, with energy: {energy}")

        states.append(state)
        energies.append(energy)
    
    # Convert to numpy arrays for easier handling
    states = np.array(states)
    energies = np.array(energies)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_states), energies, 'b-', label='Energy')
    plt.axvline(optimal_index, ymin=min(energies)-10, ymax=max(energies)+10, color='r', linestyle='-', label=f'Optimal State: {optimal_energy:.2f}')
    plt.axvline(received_index, ymin=min(energies)-10, ymax=max(energies)+10, color='y', linestyle='--', label=f'Received State: {received_energy:.2f}')
    plt.axhline(y=energies[optimal_index], color='g', linestyle='--', label='Energy of Optimal State')
    plt.xlabel('State Index')
    plt.ylabel('Energy')
    plt.title(f'Energy Landscape for {N} Variables')
    plt.grid(True)
    plt.legend()
    plt.savefig("./energy_landscape.png")
    plt.close()

def energy_original(Htilde, ytilde, xtilde):
    energy = ytilde - Htilde @ xtilde
    energy = np.power(np.linalg.norm(energy, ord=2), 2)
    return energy

def energy_Ising(Htilde, ytilde, M, sigma:np.ndarray):
    r = int(np.ceil(np.log2(np.sqrt(M))))
    N = np.shape(ytilde)[0]
    
    T = np.block([2**(r-i)*np.eye(N) for i in range(1, r+1)])
    constant = ytilde - Htilde @ (T @np.ones((r*N,)) - (np.sqrt(M) - 1) * np.ones((N,)))
    bias = 2*(ytilde - Htilde@(T@np.ones((r*N,))-(np.sqrt(M)-1)*np.ones((N,))))
    bias = bias.T @ Htilde @ T
    coupling =  -2*T.T @ Htilde.T @ Htilde @ T 
    diagonal = np.copy(np.diag(coupling))
    np.fill_diagonal(coupling, 0)
    energy = constant.T@constant - bias.T @ sigma - 1/2*sigma.T@coupling@sigma -1/2*np.sum(diagonal)
    return energy

def Hamiltonian_test():
    N = 4
    M = 16
    r = int(np.ceil(np.log2(np.sqrt(M))))
    SNR = 10
    seed = 2

    H, symbols = MU_MIMO(N, N, M)
    Htilde = np.block([[np.real(H), -np.imag(H)], [np.imag(H), np.real(H)]])
    x = np.random.choice(symbols, (N,)) + 1j* np.random.choice(symbols, (N,))
    xtilde = np.reshape(np.block([np.real(x), np.imag(x)]), (-1, 1))
    model, xtilde, ytilde = MIMO_to_Ising(H, x, SNR, N, N, M, 2)
    sigma = decode(M, xtilde)

    LOGGER.info("================== Optimal solution ==================")
    energy_orig = energy_original(Htilde, ytilde, xtilde)
    energy_orig_ising = energy_Ising(Htilde, ytilde, M, sigma)
    energy_model = model.evaluate(sigma)
    LOGGER.info(f"Energy of original Hamiltonian: {energy_orig}")
    LOGGER.info(f"Energy of original Ising Hamiltonian: {energy_orig_ising}")
    LOGGER.info(f"Energy of Ising model: {energy_model}")

    LOGGER.info("================== Random solution ==================")
    x = np.random.choice(symbols, (N,)) + 1j* np.random.choice(symbols, (N,)) 
    xtilde = np.reshape(np.block([np.real(x), np.imag(x)]), (-1, ))
    sigma = decode(M, xtilde)

    energy_orig = energy_original(Htilde, ytilde, xtilde)
    energy_orig_ising = energy_Ising(Htilde, ytilde, M, sigma)
    energy_model = model.evaluate(sigma)
    LOGGER.info(f"Energy of original Hamiltonian: {energy_orig}")
    LOGGER.info(f"Energy of original Ising Hamiltonian: {energy_orig_ising}")
    LOGGER.info(f"Energy of Ising model: {energy_model}")



if __name__ == "__main__":
    Hamiltonian_test()