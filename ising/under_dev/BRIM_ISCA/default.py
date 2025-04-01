from dataclasses import dataclass

@dataclass
class params:
    tstart: float = 0.
    tstop: float = 1.1e-5
    tstep: float = 2.2e-11
    Rc: float = 31e3
    R: float = 31e3
    C: float = 49e-15
    Kap: float = tstop/4
    sh_it_frac: float = 2e-4
    sh_R :float= 1e3
    p0: float = 0.001799
    p1: float = 2e-6
    steps: int = int((tstop - tstart) / tstep)
    anneal_type: bool = False 
    seed: int = 1
    sh_enable: bool = False
    debug: bool = True
    cnt_flips: bool = True
    dump_fl: bool = False
    cnt_loc_min: bool = False