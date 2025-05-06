from dataclasses import dataclass

@dataclass
class params:
    tstart: float = 0
    tstop: float = 1.1e-05
    tstep: float = 2.2e-11
    Rc: float = 31e3
    R: float = 31e3
    C: float = 49e-15
    Kap: float = tstop / 4
    sh_it_frac: float = 2e-04
    sh_R: float = 1e3
    p0: float = 0.001799
    p1: float = 2e-06
    steps: int = round((tstop-tstart)/tstep)
    anneal_type: int = 1
    seed: int = 1234
    sh_enable: bool = False
    debug: bool = False
    bias: bool = False
    cnt_flips: bool = True
    dump_fl: bool = False
    cnt_loc_min: bool = False
    stop_criterion: float = 1e-16
    sf_freq: int = 1