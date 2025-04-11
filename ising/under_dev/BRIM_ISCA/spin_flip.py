import numpy as np
from numpy.random import MT19937, Generator
from dataclasses import dataclass
from typing import Callable

from ising.flow import LOGGER
from ising.under_dev.BRIM_ISCA.default import params as par


@dataclass
class SpinFlip:
    tot_sfs: int = 0
    count: int = 0
    
    def __init__(self, num_nodes: int, params: par):
        self.tot_sfs = 0
        self.count = 0
        self.sh_iters = params.sh_it_frac * params.steps
        self.p0 = params.p0
        self.p1 = params.p1
        self.p = self.p0
        self.scale = ((self.p1 - self.p0) / float(params.steps - 1)) if params.steps > 1 else 0
        
        # Initialize Mersenne Twister generator with same seed as C++
        self.generator = Generator(MT19937(seed=params.seed))
        self.rng_: Callable = lambda size: self.generator.uniform(0, 1, size)


def do_spinflip(sf: SpinFlip, state:np.ndarray, sh_cnt:np.ndarray, sh_tv:np.ndarray, sh_ts:np.ndarray, params: par):
    # Increment count
    sh_cnt = np.where(sh_cnt != -1, sh_cnt + 1, -1)
    
    # Generate random numbers
    rnd_v = sf.rng_(state.shape)
    
    # Select nodes for spin flip
    sel_sf = (rnd_v < sf.p)
    
    if params.bias:
        sel_sf[-1] = False  # Equivalent to d.eff_nodes in C++
    
    # Apply spin flip
    absv = np.where(state > 0, 1, np.where(state < 0, -1, 0))
    sh_tv = np.where(sel_sf, -absv, sh_tv)
    sh_ts = np.where(sel_sf, True, sh_ts)
    
    # Reset counters
    sh_cnt = np.where(sel_sf, 0, sh_cnt)
    
    # Handle timeout
    time_up = (sh_cnt == sf.sh_iters)
    sh_tv = np.where(time_up, 0, sh_tv)
    sh_ts = np.where(time_up, False, sh_ts)
    sh_cnt = np.where(time_up, -1, sh_cnt)
    
    # Update statistics
    sf.tot_sfs += np.sum(sel_sf)
    sf.p += sf.scale
    
    return sh_cnt, sh_tv, sh_ts
