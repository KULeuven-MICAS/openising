import pathlib
import numpy as np

from ising.model.ising import IsingModel
from ising.solvers.BRIM import BRIM
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.SA import SASolver
from ising.solvers.DSA import DSASolver


def run_solver(
    solver: str,
    num_iter: int,
    s_init: np.ndarray,
    model: IsingModel,
    clock_freq: float=1e6,
    clock_op: int=1000,
    logfile: pathlib.Path | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, float]:
    """Solves the given problem with the specified solver.

    Args:
        solver (str): The solver
        num_iter (int): amount of iterations
        s_init (np.ndarray): initial state
        model (IsingModel): model to use
        logfile (pathlib.Path | None, optional): path to logfile to store data. Defaults to None.

    Returns:
        tuple[np.ndarray, float]: optimal state and energy of the specified solver.
    """
    optim_state = np.zeros((model.num_variables,))
    optim_energy = None
    if solver == "BRIM":
        v = 0.1 * np.ones((model.num_variables,)) * s_init
        optim_state, optim_energy = BRIM().solve(
            model=model,
            v=v,
            num_iterations=num_iter,
            dt=hyperparameters["dtBRIM"],
            kmin=hyperparameters["kmin"],
            kmax=hyperparameters["kmax"],
            C=hyperparameters["C"],
            G=hyperparameters["G"],
            file=logfile,
            random_flip=hyperparameters["flip"],
            seed=hyperparameters["seed"],
        )
    elif solver == "SA":
        optim_state, optim_energy = SASolver().solve(
            model=model,
            initial_state=s_init,
            num_iterations=num_iter,
            initial_temp=hyperparameters["T"],
            cooling_rate=hyperparameters["r_T"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver == "DSA":
        optim_state, optim_energy = DSASolver().solve(
            model=model,
            initial_state=s_init,
            num_iterations=num_iter,
            initial_temp=hyperparameters["T"],
            cooling_rate=hyperparameters["r_T"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver == "SCA":
        optim_state, optim_energy = SCA().solve(
            model=model,
            sample=s_init,
            num_iterations=num_iter,
            T=hyperparameters["T"],
            r_t=hyperparameters["r_T"],
            q=hyperparameters["q"],
            r_q=hyperparameters["r_q"],
            seed=hyperparameters["seed"],
            file=logfile,
            clock_freq=clock_freq, clock_op=clock_op,
        )
    elif solver[1:] == "SB":
        x = 0.01*np.ones((model.num_variables,))*s_init
        y = np.zeros((model.num_variables,))
        dt = hyperparameters["dtSB"]
        at = hyperparameters["at"]
        c0 = hyperparameters["c0"]
        a0 = hyperparameters["a0"]
        if solver[0] == "b":
            optim_state, optim_energy = ballisticSB().solve(
                model=model,
                x=x,
                y=y,
                num_iterations=num_iter,
                at=at,
                c0=c0,
                dt=dt,
                a0=a0,
                file=logfile,
                clock_freq=clock_freq, clock_op=clock_op,
            )
        elif solver[0] == "d":
            optim_state, optim_energy = discreteSB().solve(
                model=model,
                x=x,
                y=y,
                num_iterations=num_iter,
                at=at,
                c0=c0,
                dt=dt,
                a0=a0,
                file=logfile,
                clock_freq=clock_freq, clock_op=clock_op,
            )
    return optim_state, optim_energy


def return_rx(num_iter: int, r_init:float, r_final:float):
    return (r_final/r_init)**(1/(num_iter + 1))

def return_c0(model: IsingModel):
    return 0.5 / (
        np.sqrt(model.num_variables)
        * np.sqrt(np.sum(np.power(model.J, 2)) / (model.num_variables * (model.num_variables - 1)))
    )
