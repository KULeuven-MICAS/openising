from ising.model import BinaryQuadraticModel
import numpy as np

class runSolver:
    """
    This class can solve a given problem on with a variety of solvers.
    The currently implemented solvers are:\n
    - Simulated annealing (SA)
    - Stochastic Cellular Automata Annealing (SCA)
    - Simulated Bifurcation:
        - ballistic Simulated Bifurcation (bSB)
        - discrete Simulated Bifurcation (dSB)
    - Bistable Resistively-coupled Ising Machine (BRIM)\n
    There is a possibility to solve the problem multiple with multiple solvers
    in order to compare them on a given attribute. Or you can also solve the problem with only one solver.

    """
    def __init__(self, problem: BinaryQuadraticModel):
        """Initializes the solver runner.

        Args:
            problem (BinaryQuadraticModel): the specific problem that needs to be solver
        """
        self.problem = problem
        self.solvers = ['SA', 'SCA', 'bSB', 'dSB', 'BRIM']
        self.energies = {}
        self.sigma = {}
        self.current_run = ''

    def run_step_test(self, nb_runs:int, Nstep_list: list[int]):
        """Runs a test for different iteration lenghts.
        Every iteration length will be run nb_runs times.


        Args:
            nb_runs (int): number of runs per step size
            Nstep_list (list[int]): all the step lengths
        """

    def run_Tcomp_test(self):
        """
        Runs a test for different computation time lengths.
        """

    def run_solver(self):
        """
        Runs a single solver with the given hyperparameters.
        """

    def __get_random_s__(self):
        """
        Creates a random sample of spins.

        :return s (np.ndarray): random vector of spins
        """
        return np.random.choice([-1, 1], self.problem.num_variables)

    def compute_energy(self):
        """
        Computes the Hamiltonian given a sample.
        """

    def plot_energies(self):
        """
        Plots the energies of a all the solvers in the current run
        """

