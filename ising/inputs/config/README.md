Config format
=============

The config file is written in YAML. It must has the following parameters:

*benchmark:* [str] the workload file name, corresponding to the problem type set in API.

*iter_list:* [positive int] also called as trail length, the number (integer) of iterations to run in the solver. Multiple values can be defined so that solvers will run with different trail length.

*solvers:* [str] solvers to run. Options include: BRIM, SA, bSB, dSB, SCA, Multiplicative, all

*nb_runs:* [positive int] number (integer) of trials to run.

*use_gurobi:* use local gurobi to simulate if True, otherwise use local solver if False. This will override *solvers*.

*use_multiprocessing:* not used yet.

### Following parameters are optional, depending on the solvers used.

**Parameters for SA solver**

*T:* [float] initial temperature for the annealing solvers (SA and SCA).

*T_final:* [float] final temperature, which should be lower than *T*, for the annealing solvers (SA and SCA).

*seed:* [float] the seed used for random number generation. This is important to be able to recreate results.

**Parameters for SCA solver**

*T:* [float] initial temperature for the annealing solvers (SA and SCA).

*T_final:* [float] final temperature, which should be lower than *T*, for the annealing solvers (SA and SCA).

*q:* [float] the coupling strength between the two states in the SCA solver. When this value is 0, the most optimal one is chosen and it is not annealed.

*q_final:* [float] the final coupling strength between the two states in the SCA solver.

**Parameters for Multiplicative solver**

*dtMult:* [float] time step for the Multiplicative solver.

*T_cont:* [float] will remove in branch

*nb_flipping:* [int] amount of times flipping will be done. 

*cluster_threshold:* [float] threshold value for designing the cluster.

*init_cluster_size:* [float] the beginning cluster size for flipping. This value is a float between 0 and 1.

*end_cluster_size:* [float] final cluster size for flipping.This value is a float and between 0 and 1.

*T_final_cont:* [float] will remove in branch

*resistance:* [float] the resistance used in Multiplicative solver. Default value is 1.

*capacitance:* [float] the capacitance.

*nb_flipping:* amount of times flipping will be done. 

*cluster_threshold:* threshold value for designing the cluster.

*init_cluster_size:* the beginning cluster size for flipping. This value is a float between 0 and 1.

*end_cluster_size:* final cluster size for flipping.This value is a float and between 0 and 1.

**Parameters for BRIM solver**

*dtBRIM:* [float] time step used for the BRIM solver.

*stop_criterion:* [float] smallest possible change of the voltages to mark convergence in the Multiplicatve solver.

**Parameters for SB (bSB/dSB) solver**

*dtSB:* [float] the time step used in the Simulated Bifurcation solvers (dSB and bSB).

*a0:* [float] the bifurcation parameter to which a(t) will converge to. Defaults to 1.

*c0:* [float] the parameter that defines the strength of the Ising part in the solver. When it is set to 0 will the optimal parameter be used.

### Following parameters are required only when the targeted benchmark is TSP.

*weight_constant:* penalty value added to the constraints in the TSP formulation.

### Following parameters are required only when the targeted benchmark is MIMO.

*SNR:* the Signal Noise Ratio value (integer) at which the MIMO problem is going to be solved. Multiple values can also be given.

*Nt:* [positive int] amount of user antennas for the MIMO problem.

*Nr:* [positive int] amount of receiver antennas for the MIMO problem.

*M:* [2, 4, 8, et.al.] the modulation scheme used, i.e. M-QAM, for the MIMO problem.

*nb_trials*: [positive int] amount of symbols each user needs to send. More means the BER will be more correct.

## Extra note

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

Besides, the following parameters will be added within returned ans:

*quantized_model:* [IsingModel] the Ising model after quantization.

*original_required_int_precision:* [int] the J precision required in the Ising model without quantization (h is not quantized).
