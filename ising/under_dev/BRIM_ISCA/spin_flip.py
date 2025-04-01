import numpy as np
from dataclasses import dataclass

from ising.under_dev.BRIM_ISCA.default import params as par


@dataclass
class SpinFlip:
    tot_sfs: int = 0
    
    count: int = 0

    sh_iters: int = par.sh_it_frac * par.steps
    p0: float = par.p0
    p1: float = par.p1
    p: float = p0
    
    scale: float  = ((p1 - p0) / float(par.steps - 1))  if (par.steps > 1) else 0
    inf: float    = float(np.inf)

    generator: np.random.Generator = np.random.Generator(np.random.MT19937())

    # For custom Spin flip


def do_spinflip(sf: SpinFlip, state:np.ndarray, sh_cnt:np.ndarray, sh_tv:np.ndarray, sh_ts:np.ndarray):
    sh_cnt = np.where((sh_cnt != -1), sh_cnt + 1, -1)

    rnd_v = sf.generator.random(state.shape)

    sel_sf = (rnd_v < sf.p)

    absv = np.where(state > 0, 1, -1)
    sh_tv = np.where(sel_sf, -absv, sh_tv)
    sh_ts = np.where(sel_sf, True, sh_ts)

    sh_cnt = np.where(sel_sf, 0, sh_cnt)

    time_up = (sh_cnt == sf.sh_iters)
    sh_tv = np.where(time_up, 0, sh_tv)
    sh_ts = np.where(time_up, False, sh_ts)
    sh_cnt = np.where(time_up, -1, sh_cnt)

    sf.tot_sfs += np.sum(sel_sf)

    sf.p += sf.scale
