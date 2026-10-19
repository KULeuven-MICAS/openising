import numpy as np

from numba import njit

from ising.solvers.Multiplicative import Multiplicative
from ising.stages.model.ising import IsingModel
from ising.utils.HDF5Logger import HDF5Logger


@njit(cache=True)
def _time_to_zero(v, s, f):
    """
    Time until a voltage reaches zero.
    """
    if v * f < 0 or (v == 0.0 and s * f < 0):
        return -v / f
    return np.inf


@njit(cache=True)
def _inner_loop_event_numba(coupling, state, frozen, s_ref, f_ref, t_max):
    """
    Event-based multiplactive solver that computes the slope of all sigmas, their intersection
    with the threshold and then skips to that "event".

    Between flips all the voltage change for the sigmas is linear.

    @type coupling: np.ndarray
    @param coupling: coupling matrix. MUST be symmetric.
    @type state: np.ndarray
    @param state: starting voltages
    @type frozen: np.ndarray
    @param frozen: mask that freezes nodes so they get no current
    @type s_ref: np.ndarray
    @param s_ref: best sigmas found so far
    @type f_ref: np.ndarray
    @param f_ref: fields for those best sigmas
    @type t_max: float
    @param t_max: time limit of a run
    """
    n = state.shape[0]                          # number of spins
    v = state.astype(np.float64)                # starting voltages
    s = np.empty(n)                             # signs
    for j in range(n):                          # a voltage at the threshold counts as +1
        s[j] = 1.0 if v[j] >= 0 else -1.0

    f = f_ref.copy()                            # The "local field" for the best spin state found
    for k in range(n):                          # Iterate all spins
        if s[k] != s_ref[k]:                    # spin k is different from the best state
            d = s[k] - s_ref[k]                 # difference (+-2 for flip)
            for j in range(n):                  # update local fields related to the flip
                if not frozen[j]:
                    f[j] += coupling[k, j] * d

    t_next = np.inf                             # time when the next spin crosses the threshld
    k = -1                                      # index of first spin that crosses the threshold
    for j in range(n):
        t_j = _time_to_zero(v[j], s[j], f[j])   # find the spin the crosses first
        if t_j < t_next:
            t_next = t_j
            k = j

    t = 0.0                                     # time since start
    while k >= 0:                               # is there a spin thats heading for the threshold
        if t + t_next > t_max:
            return s, f, t_max

        # Flip one spin per event
        d = np.sign(f[k]) - s[k]                # difference (+-2 for flip)
        s[k] = np.sign(f[k])                    # update the sign
        t += t_next                             # jump to next crossing

        # update all voltages and find the next crossing
        # continue until no spin is heading for the threshold (k = -1)
        dt_event = t_next                                               # time step till next crossing
        k_flip = k                                                      # flip that crosses at t_next
        t_next = np.inf                                                 # reset t_next, we will find the next
        k = -1                                                          # reset k, we will find the next
        for j in range(n):
            reached = v[j] * f[j] < 0 and -v[j] / f[j] <= dt_event      # is the spin headed for the threshold and
                                                                        # will it pass the threshold within dt_event ?
            v[j] = min(1.0, max(-1.0, v[j] + dt_event * f[j]))          # update the voltage after dt_event
            if reached:
                v[j] = 0.0                                              # if the spin crossed thr, force v to 0.0
            if not frozen[j]:
                f[j] += coupling[k_flip, j] * d                       # update local fields from flipped spin
            t_j = _time_to_zero(v[j], s[j], f[j])                       # when does this spin cross the threshold
            if t_j < t_next:                                            # save if earliest crossing found so far
                t_next = t_j
                k = j

    # Time until every voltage sits at a rail
    t_rail = 0.0
    for i in range(n):
        if v[i] != s[i]:
            if s[i] * f[i] > 0:
                t_rail = max(t_rail, (s[i] - v[i]) / f[i])
            else:
                t_rail = np.inf
    return s, f, min(t + t_rail, t_max)


class MultiplicativeEvent(Multiplicative):
    """
    Multiplicative solver that replaces the Forward Euler loop by a time-skipping mechanism.

    Is based on Multiplicative, so it inherits most of the functionality.

    Inherited:
        - inner_loop_FE: just the hook name. No Forward Euler is done.
        - stop_criterion: not used since convergence is exact
        - dt, num_iterations: only used for setting the time limit t_max
        - tau_system: only used so the op count matches
        - total_delay: does not work
        - logging: ignored
    """

    def __init__(self, adjustments=False):
        super().__init__(adjustments)
        self.name = "MultiplicativeEvent"

    def set_params(self, *args, **kwargs):
        super().set_params(*args, **kwargs)                             # run Multiplactive's set_params

        self.frozen = np.zeros(self.coupling.shape[0], dtype=np.bool_)  # just build a boolean mask
        self.frozen[self.freeze_nodes] = True
        if self.bias:
            self.frozen[-1] = True

        self.dv_scale = self.current / self.capacitance                 # used for t_max
        self.t_max = self.num_iterations * float(self.dt) * self.dv_scale
        self.time_per_ops = max(1, int(np.ceil(self.tau_system / self.dt))) * float(self.dt)

        self.cache_energy = np.inf
        self.cache_s = np.zeros(self.coupling.shape[0])                 # cache the sigmas
        self.cache_f = np.zeros(self.coupling.shape[0])                 # cache the local fields

    def inner_loop_FE(
        self,
        model: IsingModel,
        state: np.ndarray,
        total_delay: int,
        logging: HDF5Logger | None = None,
    ) -> tuple[np.ndarray, float, float, int]:
        """
        This is NOT an FE solver! Just by name due to inheritance.

        Delays are not supported and the voltages are not logged over time.

        @type model: IsingModel
        @param model: the model to solve.
        @type state: np.ndarray
        @param state: the initial state to start the simulation.
        @type total_delay: int
        @param total_delay: total delay present in the system, must be 0.
        @type logging: HDF5Logger|None
        @param logging: unused.
        @rtype: tuple[np.ndarray, float, float, int]
        @return: the new state, new energy, time until all voltages reach a rail and number of operations.
        """
        # Throw an error if total_delay is attempted to be used
        if total_delay > 0:
            raise NotImplementedError("MultiplicativeEvent does not support delays")

        # Run the solver
        s, f, t = _inner_loop_event_numba(
            self.coupling, state, self.frozen, self.cache_s, self.cache_f, self.t_max
        )
        t /= self.dv_scale
        nb_operations = int(t / self.time_per_ops) * (2 * model.num_variables**2 + 3 * model.num_variables)

        sigmas = s[: model.num_variables].astype(np.float32)
        energy = model.evaluate(sigmas)
        if energy < self.cache_energy:
            self.cache_energy, self.cache_s, self.cache_f = energy, s, f
        return sigmas, energy, t, nb_operations
