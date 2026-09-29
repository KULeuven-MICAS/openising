 [Model predictive path integral control (MPPI)](https://arxiv.org/abs/2512.15533)
 ----------------------------------------------------
> This is the benchmark suite for model predictive path integral control using ising machines.
> Reference to the paper is provided as a link in the title.
> 
 # MPPI Benchmark
 > Benchmark is formatted as a .yaml file containing the necessary parameters for scene generation and model selection.

 ## Benchmark structure
This benchmark consists of multiple components:
1. Dynamics model
* This model defines the behavior of the dynamics of the agent under actions. I.e. How the state of the agent changes under some action.  <br> 
In this code the following general dynamics are considered <br> `x[i + 1] = x + f(x, a) * dt`
* This benchmark defaults to the kinematics of the [bicycle steering model](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html) <br>
* The code for these dynamics is in `/stages/model/MPPI/dynamics.py`
2. Reference trajectory
* A reference trajectory is a set of states that the agent must visit. <br>
The goal of control problems is to find a set of actions, possible under constraints, that track this reference. Meaning visit these states given the known dynamics model.
* In this benchmark the reference trajectories are generated as splines between a set number of control points in a 2D generated environment.
* The code for reference trajectory generation is in `/stages/model/MPPI/environment.py` and an example of a scene with interpolated reference points is presented below.
Note that for optimization purposes only the reference trajectory is needed. The scene itself is unused.
![Image](control_scene.png)
5. Optimization target
* The control for MPPI is formulated as a QUBO problem requiring two matrices Q and R representing the cost of a state and the action respectively.
This defines the cost to be optimized as J = x Q x + a R a which is mapped to an ising model. 
* Additional parameters for control are also defined. More details below and in the documentation.
* The code for this QUBO control is in `/stages/model/MPPI/QUBOController.py`

## Benchmark format
The `.yaml` benchamrk file contains the necessary parameters to define the whole problem. In the following subparts and their use are presented. 

### Dynamics model
> Contains configuration parameters for the dynamics model 
- dynamics: 'Bicycle' # String selection of model from `dynamics.py`
 

### General parameters
> Contains general parameters for both dynamics model as scene generation. Both dt and velocity will influence the length of the reference trajectory generated, but not the shape.
- delta_t: .1 # Time step for Eulerian forward dynamics
- velocity: 4.0 # Velocity for generation of the splines

### Trajectory parameters
> Contains parameters for reference generation

- seed: 42 # Generation seed
- nb_control_points: 7 # Number of control points between goal and start

### QUBO parameters (for bicycle riding)
> Contains configurable parameters for QUBO problem and controller. <br>
> Full Q and R penalize cross terms between states while diagonals only penalize each dimension separately. <br>


> First two dimensions are strongly penalized which corresponds to x and y coordinates. <br>
> This results in free moving steering coordinates while tracking the reference in physical space.

* Q: null   # No full Q
* R: null   # Nu full R
* Q_diag: [1.0, 1.0, 0.001, 0.0, 0.0]      # Default Q diagonal
* R_diag: [1., 1.]                         # Default R diagonal


> Slack weight is defined to balance Q and R terms, while the terminal weight adds additional importance to the final point of the rollout.
> This can be used to firmly anchor the rollout on the trajectory.
- Q_slack: 177.           # Slack weight for Q matrix
- TERMINAL_WEIGHT: 1.     # Weight for terminal state

> Dimensions of the state and action space 
* state_dim: 5            # State space dimension     --> compatible with Q
* action_dim: 2           # Action space dimension    --> compatible with R

> Dimensions and multipliers for encoding the action space into binary
> Additional information is found in the documentation.
* K: 1.                   # Multiplier for bits
* N_BITS: 16              # Number of bits
* action_scales: null     # Optional action scale
* scale_multiplier: null  # Optional multiplier


### MPC parameters
> Contains parameters for model predictive control
> Length of rollout (amount of optimized steps) and amount of actions chosen from this optimization to execute before doing new optimization.
* HORIZON_LENGTH: 8       # Planning horizon length
* action_horizon: 1       # Action execution horizon -> should be one for receding horizon

### MPPI parameters
> Contains specific parameters for the MPPI solver that are not the ising solver parameters.
> Amount of internal iterations and amount of approximations for matrix inversion during model construction. Both these parameters are explained in documentation.

* n_mppi_iterations: 1    # Number of MPPI iterations to do per tracking step
* n_approx_iter: 2        # Number of approximation terms in the calculation of inverses during forward rollout

