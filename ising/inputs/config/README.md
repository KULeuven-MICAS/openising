# Config format

The config file is written in YAML. It must has the following parameters:

**benchmark:** the workload file name, corresponding to the problem type set in API.

**iter_list:** also called as trail length, the number (integer) of iterations to run in the solver. Multiple values can be defined so that solvers will run with different trail length.

**solvers:** solvers to run. Options include: BRIM, SA, bSB, dSB, SCA, Multiplicative, all

**nb_runs:** number (integer) of trials to run.

**use_gurobi:** use local gurobi to simulate if True, otherwise use local solver if False. This will override **solvers**.

**use_multiprocessing:** not used yet.

**weight_constant:** penalty value added to the constraints in the TSP formulation.

**SNR:** the Signal Noise Ratio value (integer) at which the MIMO problem is going to be solved. Multiple values can also be given.

**Nt:** amount of user antennas for the MIMO problem.

**Nr:** amount of receiver antennas for the MIMO problem.

**M:** the modulation scheme used, i.e. M-QAM, for the MIMO problem.

**dtMult:** time step for the Multiplicative solver.

**T_cont:** will remove in branch

**T_final_cont:** will remove in branch

**coupling_annealing:** whether to anneal the coupling in the Multiplicative solver.

**resistance:** the resistance used in Multiplicative solver. Default value is 1.

**flipping:** whether to turn on flipping or not. 

**flipping_freq:** at what frequency the flipping should take place.

**flipping_prob:** the beginning probability of flipping acceptance.

**mu_param:** parameter that defines the strength of the diode in the Multiplicative solver. When negative (positive), nodes are pushed away from (attracted to) -1 and 1.

**capacitance:** the capacitance 

**dtBRIM:** time step used for the BRIM solver.

**C:** NA (removed in my branch)

**stop_criterion:** smallest possible change of the voltages to mark convergence in the Multiplicatve solver.

**noise:** NA (removed in my branch)

**T:** initial temperature for the annealing solvers (SA and SCA).

**T_final:** final temperature, which should be lower than **T**, for the annealing solvers (SA and SCA).

**seed:** the seed used for random number generation. This is important to be able to recreate results.

**q:** the coupling strength between the two states in the SCA solver. When this value is 0, the most optimal one is chosen and it is not annealed.

**q_final:** the final coupling strength between the two states in the SCA solver.

**dtSB:** the time step used in the Simulated Bifurcation solvers (dSB and bSB).

**a0:** the bifurcation parameter to which a(t) will converge to. Defaults to 1.

**c0:** the parameter that defines the strength of the Ising part in the solver. When it is set to 0 will the optimal parameter be used.