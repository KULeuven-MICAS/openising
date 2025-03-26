import numpy as np
import helper_functions as hf


def discreteSB(h:np.ndarray, J:np.ndarray, x_init:np.ndarray, y_init:np.ndarray, dt:float, Nstep:int, a0:float, c0:float, at:callable, verbose:bool=False):
    """
    Implements discrete Simulated bifurcation as is seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953). 
    This implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm.

    :param np.ndarray h: magnetic field coefficients
    :param np.ndarray J: interaction coefficients
    :param np.ndarray x_init: initial position
    :param np.ndarray y_init: initial momentum
    :param float dt: time step
    :param int Nstep: total amount of steps undertaken
    :param float a0,c0: positive constants
    :param callable at: control parameter that increases from zero
    :return sigma (np.ndarray): optimal spin configuration
    :return energies (list): energies during the optimisation
    :return time (list): time during simulation
    """
    x = np.copy(x_init)
    y = np.copy(y_init)
    N = np.shape(h)[0]
    energies = []
    times = []
    tk = 0
    if verbose:
        header = ['Time step', 'Energy value']
        print("{: >20} {: >20}".format(*header))
    for i in range(Nstep):
        for j in range(N):
            y[j] += (-(a0 - at(tk))*x[j] + c0*np.inner(J[:, j], np.sign(x)) + c0*h[j])*dt
            x[j] += a0*y[j]*dt
            if np.abs(x[j]) > 1:
                x[j] = np.sign(x[j])
                y[j] = 0
        times.append(tk)
        energy = hf.compute_energy(J=J, h=h, sigma=np.sign(x))
        if verbose:
            row = [tk, energy]
            print("{: >20} {: >20}".format(*row))
        energies.append(energy)
        tk += dt
    sigma = np.sign(x)
    return sigma, energies, times


def ballisticSB(h:np.ndarray, J:np.ndarray, x_init:np.ndarray, y_init:np.ndarray, dt:float, Nstep:int, a0:float, c0:float, at:callable, verbose:bool=False):
    """
    Implements the ballistic Simulated Bifurcation algorithm as seen in the paper of [Goto et al.](https://www.science.org/doi/10.1126/sciadv.abe7953). 
    The implementation is an improved version of the classical adiabatic Simulated Bifurcation algorithm by suppressing analog errors.

    :param np.ndarray h: magnetic field coefficients
    :param np.ndarray J: interaction coefficients
    :param np.ndarray x_init: initial position
    :param np.ndarray y_init: initial momentum
    :param float dt: time step
    :param int Nstep: total amount of steps undertaken
    :param float a0,c0: positive constants
    :param callable at: control parameter that increases from zero
    :return sigma (np.ndarray): optimal spin configuration
    :return energies (list): energies during the optimisation
    :return times (list): time during simulation
    """
    x = np.copy(x_init)
    y = np.copy(y_init)
    N = np.shape(h)[0]
    energies = []
    times = []
    tk = 0
    if verbose:
        header = ['Time step', 'Energy value']
        print("{: >20} {: >20}".format(*header))
    for i in range(Nstep):
        for j in range(N):
            y[j] += (-(a0 - at(tk))*x[j] + c0*np.inner(J[:, j], x) + c0*h[j])*dt
            x[j] += a0*y[j]*dt
            if np.abs(x[j]) > 1:
                x[j] = np.sign(x[j])
                y[j] = 0
        times.append(tk)
        energy = hf.compute_energy(J=J, h=h, sigma=np.sign(x))
        if verbose:
            row = [tk, energy]
            print("{: >20} {: >20}".format(*row))
        energies.append(energy)
        tk += dt
    sigma = np.sign(x)
    return sigma, energies, times
