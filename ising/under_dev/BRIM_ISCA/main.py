import numpy as np
import logging

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers import G_parser
from ising.generators.MaxCut import MaxCut
from ising.under_dev.BRIM_ISCA.default import params
from ising.under_dev.BRIM_ISCA.brim_isca import brim_isca
from ising.under_dev.BRIM_ISCA.multiplicative_own import multiplicative_own
from ising.utils.flow import make_directory 
from ising.utils.numpy import triu_to_symm
from ising.postprocessing.energy_plot import plot_energies_multiple
from ising.utils.HDF5Logger import return_data

fig_folder = TOP / "ising/under_dev/BRIM_ISCA/figures"
make_directory(fig_folder)

def main():
    parameters = params()

    if parameters.debug:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    LOGGER.info("parsing K2000 benchmark")
    graph_K2000 = G_parser(TOP / "ising/benchmarks/G/K2000.txt")
    LOGGER.info(f"Best found solution: {-graph_K2000[1]}")
    model = MaxCut(graph_K2000[0])
        
    initial_state = np.loadtxt(TOP / "ising/flow/000.txt")[:2000]

    cond = np.linalg.cond(triu_to_symm(model.J))
    eig,_ = np.linalg.eig(triu_to_symm(model.J))

    LOGGER.info(f"Condition number and largest eigenvalue of J are: {cond}, {np.max(np.abs(eig))}")
    logfiles = []
    LOGGER.info(parameters)
    logfile = TOP / f"ising/under_dev/BRIM_ISCA/logs/K2000_isca{"_coupling" if parameters.anneal_type else ""}{"_flipping" if parameters.sh_enable else ""}.log"
    logfiles.append(logfile)
    # state, energy = brim_isca(model, 
    #                           initial_state, 
    #                           logfile, 
    #                           parameters)
    # LOGGER.info(f"Best energy: {energy} with state: {state}")
    
    LOGGER.info("Setting up own solver")
    resistance = parameters.Rc
    capacitance = parameters.C
    tau_own = resistance * capacitance
    tau_ISCA = parameters.Rc * parameters.C
    alpha = tau_own / tau_ISCA
    tend = alpha * parameters.tstop
    dtMult = parameters.tstep
    num_iterations = int(tend / dtMult)
    ZIV = True
    coupling_annealing = True
    flipping = False

    LOGGER.info(f" resistance: {resistance}, capacitance: {capacitance}, tau_own: {tau_own}, tau_ISCA: {tau_ISCA}, alpha: {alpha}, tend: {tend}, dtMult: {dtMult}, num_iterations: {num_iterations}")

    logfile = TOP / f"ising/under_dev/BRIM_ISCA/logs/K2000_own_alpha{alpha:.2f}{"_coupling" if coupling_annealing else ""}{"_flipping" if flipping else "_noise"}{"_own_ZIV2" if ZIV else ""}.log"
    logfiles.append(logfile)

    # lam, _ = np.linalg.eig(triu_to_symm(model.J)/(resistance*capacitance))

    # LOGGER.info(f"Largest eigenvalue of system is: {np.max(lam)}")

    state, energy = multiplicative_own(model, initial_state, dtMult, 
                                       num_iterations,
                                       resistance, capacitance,
                                       parameters.seed,
                                       1e-3, 1e-7, 
                                       stop_criterion=1e-10, 
                                       coupling_annealing=coupling_annealing, ZIV=ZIV, flipping=flipping,
                                       file=logfile)
    LOGGER.info(f"Best energy: {energy} with state: {state}")

    plot_energies_multiple(logfiles, 
                           f"comparison_ISCA_own{"_coupling" if parameters.anneal_type else ""}{"_flipping" if parameters.sh_enable else ""}{"_own_ZIV2" if ZIV else ""}_alpha{alpha:.2f}.png", 
                           save_folder=fig_folder,
                           best_found=graph_K2000[1])
    
    # en_isca = return_data(logfiles[0], "energy")
    # en_own = return_data(logfiles[1], "energy")

    # en_diff = np.linalg.norm(en_isca - en_own) / np.linalg.norm(en_isca)
    # LOGGER.info(f"Difference in energy: {en_diff}")
    
    # See if results are the same as the C++ code
    # cpp_log = TOP / "no_backup/logs_BRIM_code"
    # if parameters.anneal_type:
    #     if parameters.sh_enable:
    #         cpp_log /= "with_annealing_BRIM_code_de_5.log"
    #     else:
    #         cpp_log /= "only_coupling_BRIM_code_ode_5.log"
    # else:
    #     if parameters.sh_enable:
    #         cpp_log /= "only_flipping_BRIM_code_ode_5.log"
    #     else:
    #         cpp_log /= "no_annealing_BRIM_code_ode_5.log"
    
    # voltages_cpp = return_data(cpp_log, "voltages")
    # voltages_py = return_data(logfile, "voltages")

    # diff = np.linalg.norm(voltages_cpp[:parameters.steps, :] - voltages_py) / np.linalg.norm(voltages_cpp)
    # LOGGER.info(f"Difference in voltages: {diff}")

    # energy_cpp = return_data(cpp_log, "energy")
    # energy_py = return_data(logfile, "energy")

    # diff = np.linalg.norm(energy_cpp[:parameters.steps] - energy_py) / np.linalg.norm(energy_cpp)
    # LOGGER.info(f"Difference in energy: {diff}")


def plot_original_logs():
    logtop = TOP / "ising/under_dev/BRIM_ISCA/logs"

    logfiles = [logtop / f"K2000_own_alpha1.00_{file}" for file in ["flipping_own_ZIV2.log"]]
    logfiles.append(logtop / f"K2000_isca_flipping.log")
    plot_energies_multiple(logfiles, figName="comparison_flipping_own_ZIV2_alpha1.png", save_folder=fig_folder, best_found=-33337.0)

if __name__ == "__main__":
    # main()
    plot_original_logs()