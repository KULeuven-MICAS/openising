import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Generate a random symmetric J matrix with J_ij bounded by (-1, 1)
def generate_random_J(N):
    J = np.random.uniform(-1, 1, (N, N))  # Random values between -1 and 1
    J = (J + J.T) / 2  # Make it symmetric
    np.fill_diagonal(J, 0)  # Set the diagonal elements to 0
    return J
# Generate random initial conditions for v0 bounded by (-1, 1)
def generate_random_v0(N):
    return np.random.uniform(-1, 1, N)
# Compute the Hamiltonian
def compute_hamiltonian(v, J):
    H = 0
    for i in range(N):
        for j in range(i+1, N):  # Sum only over i < j to avoid double counting
            H += J[i, j] * np.sign(v[i]) * np.sign(v[j])
    return -H
def k_square_wave(t, total_time):
    k_min = 0.1  # Minimum value for k
    k_max = 5.0  # Maximum value for k
    num_cycles = 10  # Number of square wave cycles
    cycle_duration = total_time / num_cycles
    # Determine if we're in the high (k_max) or low (k_min) part of the square wave
    return k_max if int(t // (cycle_duration / 2)) % 2 == 0 else k_min
# Define the system of ODEs with the constraint (-1, 1)
def dvdt(t, v, J, total_time):
    dv = np.zeros(N)
    k = k_square_wave(t,total_time)
    for i in range(N):
        coupling_sum = sum(J[i, j] * (v[i] - v[j]) for j in range(N))
        rate_of_change = (1/C) * (G * np.tanh(k * np.tanh(k * v[i])) - G * v[i] - coupling_sum)
        
        # Apply boundary conditions: if v is near -1 or 1, reduce the rate of change to prevent overflow
        if v[i] >= 1.0 and rate_of_change > 0:
            dv[i] = 0  # Stop increasing when v is at the upper limit
        elif v[i] <= -1.0 and rate_of_change < 0:
            dv[i] = 0  # Stop decreasing when v is at the lower limit
        else:
            dv[i] = rate_of_change

    return dv
# Constants
N = 256   # Example value for the size of the system
C = 1e-6  # Example value for C
G = 1e-1  # Example value for G
# Generate the random J matrix and initial state
J = generate_random_J(N)
v0 = generate_random_v0(N)
# Time span for the simulation
t_span = (0, 2e-6)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
# Solve the system of ODEs
sol = solve_ivp(dvdt, t_span, v0, args=(J, t_span[1]), t_eval=t_eval, method='RK23')
# Compute the Hamiltonian and k values at each time step
hamiltonians = []
k_values = []
for t, vi in zip(sol.t, sol.y.T):  # Loop over each time step's solution
    H = compute_hamiltonian(vi, J)
    hamiltonians.append(H)
    k_values.append(k_square_wave(t, t_span[1]))  # Get k for each time
# Plot k(t) and H(t) together
fig, ax1 = plt.subplots()

# Plot k(t) on the left y-axis
ax1.plot(sol.t, k_values, label='k(t)', color='green', linestyle='--')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('k(t)', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Create a second y-axis for the Hamiltonian
ax2 = ax1.twinx()
ax2.plot(sol.t, hamiltonians, label='Hamiltonian H(t)', color='purple')
ax2.set_ylabel('Hamiltonian H(t)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Title and legend
plt.title('Square Wave of k(t) and Hamiltonian H(t)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# plt.savefig('test_N256.png')
plt.show()