Config format
=============

The config file is written in YAML. It must has the following parameters:

*benchmark:* [str] the workload file name, corresponding to the problem type set in API.

*solvers:* [str] solvers to run. Options include: BRIM, SA, bSB, dSB, SCA, Multiplicative, all

*nb_runs:* [positive int] number (integer) of trials to run.

*use_gurobi:* use local gurobi to simulate if True, otherwise use local solver if False. This will override *solvers*.

*use_multiprocessing:* Whether to run the multiple runs with multithreading or not.

*nb_threads:* [positive int] the amount of threads used for multiprocessing.

*initialization_seed:* [int] seed to initialize the initial spin states.

*eigenvalue_start:* [bool] use the eigenvalue of the coupling matrix as a starting point. 

*gen_logfile:* [bool] whether generate HDF5 log file that records all spin updating details (Default: False if not defined.)

*logfile_discrimination:* [str | None] How the logfiles can be further discriminated against each other. If None, no discriminator will be made.

## Following parameters are optional, depending on the solvers used.
**For digital solvers**

*stop_criterion_iterations:* [bool] stops the solver when the energy doesn't change for 50 iterations.

**For each solver used**

*num_iterations_{solver_name}:* [positive int] The amount of iterations the solver performs to converge to a solution.

**Parameters for SA solver**

*T:* [float] initial temperature for the annealing solvers (in-Situ SA, SA and SCA).

*T_final:* [float] final temperature, which should be lower than *T*, for the annealing solvers (in-Situ SA, SA and SCA).

*seed:* [int] the seed used for random number generation. This is important to be able to recreate results.

**Parameters for SCA solver**

*T_init_SCA:* [float] initial temperature for the annealing solvers (in-Situ SA, SA and SCA).

*T_final:* [float] final temperature, which should be lower than *T*, for the annealing solvers (in-Situ SA, SA and SCA).

*q:* [float] the coupling strength between the two states in the SCA solver. When this value is 0, the most optimal one is chosen and it is not annealed.

*q_final:* [float] the final coupling strength between the two states in the SCA solver.

**Parameters for in-Situ SA solver**

*initial_temp_inSituSA:* [float] initial temperature for the annealing solvers. This one is different from the other annealing based solvers due to the larger temperature range.

*T_final:* [float] final temperature, which should be lower than *T*, for the annealing solvers (in-Situ SA, SA and SCA).

*nb_flips:* [int] The total amount of nodes that are allowed to flip each iteration.

*seed:* [int] the seed used for random number generation. This is important to be able to recreate results.

**Parameters for Multiplicative solver**

*nb_flipping:* [int] amount of times flipping will be done. 

*cluster_threshold:* [float] threshold value for designing the cluster.

*init_cluster_size:* [float] the beginning cluster size for flipping. This value is a float between 0 and 1.

*end_cluster_size:* [float] final cluster size for flipping.This value is a float and between 0 and 1.

*cluster_choice:* [str] The choice for the cluster. The choices are random, gradient, frequency, weighted_mean_smallest, weighted_mean_largest.

*exponent:* [float] the exponent by which the exponential function of cluster size changes.

*current:* [float] the amount of unit current flows through the cells.

*capacitance:* [float] the capacitance.

*accumulation_delay:* [float] The amount of accumulation delay present. The value represent the fraction of the time constant between two coupling nodes.

*broadcast_delay:* [float] The same as accumulation delay, but now in the boradcast dimension.

*delay_offset:* [float] Amount of delay due to the comparator as a fraction of the time constant.

<!-- *sigma_J:* [float] the amount of standard deviation present in the mismatch of the coupling unit. When -1 this is disabled in the simulation.  -->

*ode_choice:* [str] which ODE solver to perform the simulation with. Currently RK (Runge-Kutta 4) and FE (Forward Euler) are implemented.

**Parameters for BRIM solver**

*dtBRIM:* [float] time step used for the BRIM solver.

*stop_criterion:* [float] smallest possible change of the voltages to mark convergence in the Multiplicatve solver.

*coupling_annealing:* [bool] Whether to slowly anneal the coupling values.

*do_flipping:* [bool] Whether to perform the flipping.

*resistance:* [float] the resistance used in Multiplicative solver. Default value is 1.

*capacitance:* [float] the capacitance.

*probability_start*: [float] The starting probability with which to flip nodes. High means that many nodes have a higher chance to flip, i.e more nodes will be flipped.

**Parameters for SB (bSB/dSB) solver**

*dtbSB:* [float] the time step used in the ballistic Simulated Bifurcation solver.

*dtdSB:* [float] the time step used in the discrete Simulated Bifurcation solver.

*a0:* [float] the bifurcation parameter to which a(t) will converge to. Defaults to 1.

*c0:* [float] the parameter that defines the strength of the Ising part in the solver. When it is set to 0 will the optimal parameter be used.

## Following parameters are required only when the targeted benchmark is TSP.

*weight_constant:* penalty value added to the constraints in the TSP formulation.

## Following parameters are required only when the targeted benchmark is MIMO.

*SNR:* [int] the Signal Noise Ratio value (integer) at which the MIMO problem is going to be solved. Multiple values can also be given.

*nb_trials*: [positive int] amount of symbols each user needs to send. More means the BER will be more correct.

*mimo_seed*: [float] seed for random number generation when applying noise (same as dummy_seed).

*is_hamming_encoding*: [bool] if use Hamming encoding for multi-bit symbols when QAM>4. Otherwise, the binary encoding will be used.

## Extra parameters for specific stages

**If NpmosStage is used, the following parameters are required:**

*offset_type:* [str] whether to scale up the negative or positive J and h. No scaling if the type is neither. Options: negative or positive or others.

*offset_ratio:* [positive float] the scaling ratio once offset_type is negative or positive. Not used if offset_type is others.

Besides, the following parameters will be added within returned ans:

*offset_model:* [IsingModel] the Ising model with offset.

**If NoiseStage is used, the following parameters are required:**

*device_noise:* [bool] if turn on the NoiseStage. Options: True or False.

*noise_level:* [positive float] the standard deviation of the Guassian noise (mean is always at 1).

Besides, the following parameters will be added within returned ans:

*noisy_model:* [IsingModel] the Ising model with injected noise.

**If QuantizationStage is used, the following parameters are required:**

*quantization:* [bool] if turn on the QuantizationStage. OPtions: True or False.

*quantization_precision:* [positive int] the targeted quantization precision.

*scale_to_integer*: [bool] whether to scale the quantized values to integer values supported on chip.

Besides, the following parameters will be added within returned ans:

*quantized_model:* [IsingModel] the Ising model after quantization.

*original_required_int_precision:* [int] the J precision required in the Ising model without quantization (h is not quantized).

**If MismatchStage is used, the following parameter is required:**

*mismatch_std:* [float] the standard deviation present in the model. When 0.0, the mismatch is automatically turned off.

**If CombineNodesStage is used, the following parameters are required:**

*combine_nodes:* [bool] whether or not to activate the CombineNodesStage. 

*nodes_scaling:* [int] the amount of nodes each node will be split into.

**If DummyCreatorStage is used, the following parameters are required:**

*dummy_creator*: [bool] if turn on the dummy creation stage.

*dummy_seed*: [float] seed for random number generation.

**If dummy MIMO is to be generated, these parameters are required:**

*dummy_qam*: [int] QAM modulation (e.g., 2, 4, 8, 16).

*dummy_snr*: [float] Signal-to-Noise ratio (in dB).

*dummy_spacing:* [float] The spacing between the receiver antennas as a fraction of the wavelength.

*dummy_user_num*: [int] the amount of users.

*dummy_ant_num*: [int] the amount of antennas at the Base Station.

*dummy_case_num*: [int] the amount of dummy input testcases to generate.

**If dummy MaxCut is to be generated, these parameters are required:**

*dummy_quadratic:* [bool] whether to generate a problem with only 1 global optimum.

*dummy_local_optima:* [bool] whether there are local optima in the dummy problem.

**If dummy MaxCut/TSP/ATSP is to be generated, these parameters are required:**

*dummy_size*: [int] the amount of nodes (cities in TSP/ATSP).

*dummy_weight_constant*: [float] (optional, only for TSP/ATSP) the weight constant to add.

**If dummy KnapSack is to be generated, these paramters are required:**

*dummy_size*: [int] the amount of items.

*dummy_density*: [float] the problem density.

*dummy_penalty_value*: [float] the penalty value.

*dummy_bit_width*: [int] bit width of the highest profit.
