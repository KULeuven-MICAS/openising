import numpy as np
import matplotlib.pyplot as plt
import logging

from ising.flow import LOGGER
from ising.generators.MIMO import MU_MIMO, MIMO_to_Ising
from ising.solvers.Multiplicative import Multiplicative
from ising.utils.flow import return_rx, return_q

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

def Hamiltonian_test():
    N = 4
    M = 64
    r = int(np.ceil(np.log2(np.sqrt(M))))
    SNR = 100
    seed = 2

    H, symbols = MU_MIMO(N, N, M, seed)
    x = np.random.choice(symbols, size=(N,)) + 1j*np.random.choice(symbols, size=(N,))
    model, xtilde, T = MIMO_to_Ising(H, x, SNR, N, N, M, seed)
    LOGGER.info(f"Transformed input signal: {xtilde}")
    sigma = decode(M, xtilde)
    energy_sol = model.evaluate(sigma)
    LOGGER.info(f"Decoded input signal: {sigma}, with energy: {energy_sol}")
    
    initial_state = np.random.choice([-1, 1], size=(model.num_variables,))
    dtMult = 1e-3
    num_iter = 100000
    mu_param = -3.55
    flipping = True
    flipping_freq = 100
    flipping_prob = 0.1

    sigma_optim, energy = Multiplicative().solve(model, 
                                                 initial_state,
                                                 dtMult, 
                                                 num_iter, 
                                                 initial_temp_cont=0.5, 
                                                 mu_param=mu_param, 
                                                 seed=seed, 
                                                 flipping=flipping, 
                                                 flipping_freq=flipping_freq, 
                                                 flipping_prob=flipping_prob)
    x_optim = T @ (sigma_optim + np.ones((r * 2*N,))) - (np.sqrt(M) - 1) * np.ones((2*N,))
    LOGGER.info(f"Decoded optimal input signal: {sigma_optim}")
    LOGGER.info(f"Optimal input signal: {x_optim}, with optimal energy: {energy}")
    diff_list = []
    for i in range(len(sigma_optim)):
        sigma_optim[i] *= -1
        flipped_energy = model.evaluate(sigma_optim)
        diff_list.append(flipped_energy >= energy)
        sigma_optim[i] *= -1
    LOGGER.info(f"Difference list: {diff_list}")
    # plot_energy(model, sigma, sigma_optim, energy_sol)



if __name__ == "__main__":
    Hamiltonian_test()