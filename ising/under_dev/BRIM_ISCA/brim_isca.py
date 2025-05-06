import numpy as np
import pathlib

from ising.flow import LOGGER
from ising.model import IsingModel
from ising.utils.numpy import triu_to_symm
from ising.utils.HDF5Logger import HDF5Logger
from ising.under_dev.BRIM_ISCA.default import params
from ising.under_dev.BRIM_ISCA.spin_flip import SpinFlip, do_spinflip

def brim_isca(model: IsingModel,
        initial_state:  np.ndarray,
        file:           pathlib.Path|None,
        parameters: params):
    
        # Set up the time evaluations
        t_eval = np.linspace(parameters.tstart, parameters.tstop, parameters.steps)

        # Transform the model to one with no h and mean variance of J
        if np.linalg.norm(model.h) >= 1e-10:
            new_model = model.transform_to_no_h()
            parameters.bias = True
        else:
            new_model = model
        J = triu_to_symm(new_model.J) * 2 / parameters.Rc

        # Set up the bias node and add noise to the initial voltages
        N = model.num_variables
        if parameters.bias:
            v = np.block([initial_state, 1.0])
        else:
            v = initial_state

        sf = SpinFlip(N, parameters)
        sh_tv = np.zeros((N + 1*int(parameters.bias),))
        sh_ts = np.full((N + 1*int(parameters.bias),), False)
        sh_cnt = -1*np.ones((N + 1*int(parameters.bias),))

        # Schema for logging
        schema = {"time_clock": float, "energy": np.float32, "state": (np.int8, (N,)), "voltages": (np.float32, (N,))}

        # Define the system equations
        def dvdt(t: float, vt: np.ndarray, coupling: np.ndarray):
            """Differential equations for the multiplicative BRIM model.

            Args:
                t (float): time
                vt (np.ndarray): current voltages
                coupling (np.ndarray): coupling matrix J

            Returns:
                dv (np.ndarray): the change of the voltages
            """

            # Buffering
            c = np.where((vt < -1.) | (vt > 1.), 1/np.abs(vt), 1.0)

            # ZIV diode
            z = vt/parameters.Rc + ((-2.156334025305975e-05 * np.power(vt, 5)) + ( 1.017179575405042e-04 * np.power(vt, 3)) + (-2.231312342175098e-05 * vt))

            if parameters.sh_enable:
                flip = (sh_tv - vt) / parameters.sh_R
                not_sh_ts =  np.where(sh_ts, False, True)
                dv = np.where(not_sh_ts, -z + np.dot(coupling,(vt*c)), flip) / parameters.C
            else:
                dv = (-z + np.dot(coupling, vt*c)) / parameters.C 

            # # Ensure the voltages stay in the range [-1, 1]
            cond1 = (dv > 0) & (vt > 1)
            cond2 = (dv < 0) & (vt < -1)
            dv *= np.where(cond1 | cond2, 0.0, 1.)

            # # Ensure the bias node does not change
            if parameters.bias:
                dv[-1] = 0.0
            return dv

        with HDF5Logger(file, schema) as log:
            log_metadata(
                logger=log,
                name=f"BRIM_ISCA{"_coupling" if parameters.anneal_type else ""}{"_flipping" if parameters.sh_enable else ""}",
                initial_state=np.sign(v[:-1]),
                model=model,
                num_iterations=parameters.steps,
                time_step=parameters.tstep,
                coupling_annealing=parameters.anneal_type,
            )

            # Set up the simulation
            i = 0
            max_change = np.inf
            previous_voltages = np.copy(v)
            NUM_ODE_STEPS = 2
            dt = parameters.tstep / NUM_ODE_STEPS

            while i < parameters.steps and max_change > parameters.stop_criterion:
                tk = t_eval[i]

                if parameters.anneal_type:
                    coupling_ann = Ka(tk, parameters.Kap)
                else:
                    coupling_ann = 1.0

                ib_voltages = np.copy(previous_voltages)

                for _ in range(NUM_ODE_STEPS):
                    dv = dvdt(tk, previous_voltages, coupling_ann * J)
                    ib_voltages = ib_voltages + dv * dt
                    tk += dt
                    # previous_voltages = np.copy(ib_voltages)

                new_voltages = np.copy(ib_voltages)


                if parameters.sh_enable and i  % parameters.sf_freq == 0:
                    sh_cnt, sh_tv, sh_ts = do_spinflip(sf, new_voltages, sh_cnt, sh_tv, sh_ts, parameters)
                # Log everything
                sample = np.sign(new_voltages[:N])
                energy = model.evaluate(sample)
                log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])

                # Update the criterion changes
                if i > 0:
                    max_change = np.linalg.norm(new_voltages - previous_voltages, ord=np.inf) / np.linalg.norm(
                        previous_voltages, ord=np.inf
                    )
                if i % 1000 == 0:
                    LOGGER.info(
                        f"Step {i} / {parameters.steps} - Energy: {energy} - Time: {tk:.4e} - Coupling: {coupling_ann:.4f} - spin flip count: {sf.tot_sfs} - max change: {max_change:.4e} -  max absolute voltage: {np.max(np.abs(new_voltages[:N]))}"
                        )
                
                previous_voltages = np.copy(new_voltages)
                i += 1  

            if max_change < parameters.stop_criterion:
                for j in range(i, parameters.steps):
                    tk = t_eval[j]
                    log.log(time_clock=tk, energy=energy, state=sample, voltages=new_voltages[:N])      

            log.write_metadata(solution_state=sample, solution_energy=energy, total_time=t_eval[-1])
        return sample, energy


def Ka(time:float, Kap:float)->float:
        """Returns the coupling annealing term.

        Args:
            time (float): the time.
            end_time (float): the end time.
        Returns:
            Ka (float): the coupling annealing term.
        """
        return 1-np.exp(-time/Kap)

def log_metadata(logger:HDF5Logger, name,  model, initial_state, num_iterations, **kwargs):
    metadata = {
            "solver": name,
            "problem_size": model.num_variables,
            "initial_state": initial_state,
            "num_iterations": num_iterations,
            **kwargs,
        }
    logger.write_metadata(**metadata)