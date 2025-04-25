import numpy as np
import logging

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers import G_parser
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.MaxCut import MaxCut
from ising.generators.TSP import TSP, get_TSP_value
from ising.under_dev.BRIM_ISCA.default import params
from ising.under_dev.BRIM_ISCA.brim_isca import brim_isca
from ising.under_dev.BRIM_ISCA.multiplicative_own import multiplicative_own
from ising.utils.flow import make_directory 
from ising.utils.numpy import triu_to_symm
from ising.postprocessing.energy_plot import plot_energies_multiple
from ising.postprocessing.plot_solutions import plot_state_continuous
from ising.utils.HDF5Logger import return_data
from ising.flow.TSP.Calculate_TSP_energy import calculate_TSP_energy
from ising.postprocessing.TSP_plot import plot_graph_solution

fig_folder = TOP / "ising/under_dev/BRIM_ISCA/figures"
make_directory(fig_folder)
logfolder = TOP / "ising/under_dev/BRIM_ISCA/logs"

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
    logfile = logfolder / f"K2000_isca{"_coupling" if parameters.anneal_type else ""}{"_flipping" if parameters.sh_enable else ""}.log"
    logfiles.append(logfile)
    # state, energy = brim_isca(model, 
    #                           initial_state, 
    #                           logfile, 
    #                           parameters)
    # LOGGER.info(f"Best energy: {energy} with state: {state}")
    
    LOGGER.info("Setting up own solver")
    resistance = parameters.Rc*10
    capacitance = parameters.C*20
    tau_own = resistance * capacitance
    tau_ISCA = parameters.Rc * parameters.C
    alpha = tau_own / tau_ISCA
    tend = alpha * parameters.tstop
    dtMult = parameters.tstep
    num_iterations = int(tend / dtMult)
    ZIV = True
    coupling_annealing = False
    flipping = False

    LOGGER.info(f" resistance: {resistance}, capacitance: {capacitance}, tau_own: {tau_own}, tau_ISCA: {tau_ISCA}, alpha: {alpha}, tend: {tend}, dtMult: {dtMult}, num_iterations: {num_iterations}")

    logfile = logfile / f"K2000_own_alpha{alpha:.2f}{"_coupling" if coupling_annealing else ""}{"_flipping" if flipping else ""}{"_own_ZIV2" if ZIV else ""}.log"
    logfiles.append(logfile)

    # lam, _ = np.linalg.eig(triu_to_symm(model.J)/(resistance*capacitance))

    # LOGGER.info(f"Largest eigenvalue of system is: {np.max(lam)}")

    state, energy = multiplicative_own(model, initial_state, dtMult, 
                                       num_iterations,
                                       resistance, capacitance,
                                       parameters.seed,
                                       0.0, 1e-7, 
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


def only_multiplicative():
    parameters = params()

    if parameters.debug:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    LOGGER.info("parsing K2000 benchmark")
    graph_K2000 = G_parser(TOP / "ising/benchmarks/G/K2000.txt")
    LOGGER.info(f"Best found solution: {-graph_K2000[1]}")
    model = MaxCut(graph_K2000[0])
        
    initial_state = np.random.uniform(-1, 1, (model.num_variables,))
    resistance = 1
    capacitance = 1

    dt = 1.1e-4
    num_iter = int(5e5)

    coupling_annealing = False
    flipping = True
    flipping_freq = 100
    flipping_prob = 0.02
    mu = -3.55
    initial_temp = 0.0
    end_temp = 1e-5
    stop_criterion = 1e-10

    logfile = logfolder / f"K2000_own_flippingfreq_{flipping_freq}_flippingprob_{flipping_prob:.2f}_mu{mu:.2f}.log"
    multiplicative_own(model=model, initial_state=initial_state, dtMult=dt, num_iterations=num_iter, 
                    resistance=resistance, capacitance=capacitance, seed=parameters.seed,
                    initial_temp_cont=initial_temp, end_temp_cont=end_temp, 
                    stop_criterion=stop_criterion,
                    coupling_annealing=coupling_annealing, mu_param=mu, flipping=flipping, 
                    flipping_freq=flipping_freq, flipping_prob=flipping_prob,
                    file=logfile,
                    name=f"Multiplicative_own")

    plot_energies_multiple([logfile], figName="comparison_own_mu.png", best_found=graph_K2000[1], save_folder=fig_folder)

def other_benchmarks():
    parameters = params()
    resistance = 1
    capacitance = 1
    dt = 1e-6
    num_iter = 5000
    benchmark = "burma14"
    if parameters.debug:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    LOGGER.info(f"parsing {benchmark} benchmark")
    graph = TSP_parser(TOP / f"ising/benchmarks/TSP/{benchmark}.tsp")
    LOGGER.info(f"Best found solution: {graph[1]}")
    model = TSP(graph[0], weight_constant=1.5)
        
    initial_state = np.loadtxt(TOP / "ising/flow/000.txt")[:model.num_variables]

    cond = np.linalg.cond(triu_to_symm(model.J))
    eig,_ = np.linalg.eig(triu_to_symm(model.J))
    LOGGER.info(f"Condition number and largest eigenvalue of J are: {cond}, {np.max(np.abs(eig))}")
    LOGGER.info(parameters)
    logfile = logfolder / f"{benchmark}_isca{"_coupling" if parameters.anneal_type else ""}{"_flipping" if parameters.sh_enable else ""}.log"
    state, energy = multiplicative_own(model, 
                              initial_state, 
                              dt, num_iter, 
                              resistance, capacitance, parameters.seed,
                              0.0, stop_criterion=parameters.stop_criterion,
                              coupling_annealing=False, mu_param=-3.33, flipping=False,
                              file=logfile,
                              name="Multiplicative_own")
    TSP_val = get_TSP_value(graph[0], state)
    calculate_TSP_energy([logfile], graph[0], False)
    LOGGER.info(f"Best energy: {TSP_val} with state: {state}")
    plot_state_continuous(logfile, f"{benchmark}_state_isca.png", save_folder=fig_folder)
    plot_graph_solution(logfile, graph[0], f"{benchmark}_solution.png", save_folder=fig_folder)

def spin_flip_tests():
    parameters = params()
    parameters.anneal_type = 0
    parameters.sh_enable = True
    resistance = 1
    capacitance = 1
    dt = parameters.tstep
    num_iter = int(parameters.steps/10)
    flipping = True
    coupling = False
    mu_value = -3.22
    sf_freq_list = [1, 10, 100, 1000]

    benchmark = "K2000"
    if parameters.debug:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    LOGGER.info(f"parsing {benchmark} benchmark")
    graph = G_parser(TOP / f"ising/benchmarks/G/{benchmark}.txt")
    LOGGER.info(f"Best found solution: {-graph[1]}")
    model = MaxCut(graph[0])
        
    initial_state = np.loadtxt(TOP / "ising/flow/000.txt")[:model.num_variables]

    cond = np.linalg.cond(triu_to_symm(model.J))
    eig,_ = np.linalg.eig(triu_to_symm(model.J))
    LOGGER.info(f"Condition number and largest eigenvalue of J are: {cond}, {np.max(np.abs(eig))}")
    LOGGER.info(parameters)
    for sh_freq in sf_freq_list:
        logfile = logfolder / f"{benchmark}_own_flippingfreq_{sh_freq}_test.log"
        state, energy = multiplicative_own(model, 
                                initial_state, 
                                dt, num_iter,
                                resistance, capacitance, parameters.seed,
                                stop_criterion=parameters.stop_criterion, initial_temp_cont=0.0, 
                                coupling_annealing=coupling, mu_param=mu_value, flipping=flipping,
                                flipping_freq=sh_freq,
                                file=logfile,
                                name=f"Multiplicative_sf_freq_{sh_freq}")
        LOGGER.info(f"Best energy: {energy} with state: {state}")

def probability_tests():
    parameters = params()
    parameters.anneal_type = 0
    parameters.sh_enable = True
    resistance = 1
    capacitance = 1
    dt = 1e-4
    num_iter = parameters.steps
    flipping = True
    coupling = False
    mu_value = -3.22
    sf_prob_list = np.linspace(0.005, 0.0001, 10)

    benchmark = "K2000"
    if parameters.debug:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)

    LOGGER.info(f"parsing {benchmark} benchmark")
    graph = G_parser(TOP / f"ising/benchmarks/G/{benchmark}.txt")
    LOGGER.info(f"Best found solution: {-graph[1]}")
    model = MaxCut(graph[0])
        
    initial_state = np.loadtxt(TOP / "ising/flow/000.txt")[:model.num_variables]

    cond = np.linalg.cond(triu_to_symm(model.J))
    eig,_ = np.linalg.eig(triu_to_symm(model.J))
    LOGGER.info(f"Condition number and largest eigenvalue of J are: {cond}, {np.max(np.abs(eig))}")
    LOGGER.info(parameters)
    for sf_prob in sf_prob_list:
        logfile = logfolder / f"{benchmark}_own_flippingprob_{sf_prob}_test.log"

        state, energy = multiplicative_own(model, 
                                initial_state, 
                                dt, num_iter,
                                resistance, capacitance, parameters.seed,
                                stop_criterion=parameters.stop_criterion, 
                                coupling_annealing=coupling, mu_param=mu_value, flipping=flipping,
                                flipping_freq=1,
                                flipping_prob = sf_prob,
                                file=logfile,
                                name=f"Multiplicative_sf_prob_{sf_prob}")
        LOGGER.info(f"Best energy: {energy} with state: {state}")

def plot_original_logs():
    logtop = TOP / "ising/under_dev/BRIM_ISCA/logs"

    logfiles = [logtop / f"K2000_own_flippingfreq_10_flippingprob_0.001799_mu-3.55.log"]
    plot_energies_multiple(logfiles, figName="comparison_flipping_freq_own.png", save_folder=fig_folder, best_found=-33337.0)

if __name__ == "__main__":
    # main()
    only_multiplicative()
    # other_benchmarks()
    # plot_original_logs()
    # spin_flip_tests()
    # probability_tests()
    