import string
from pathlib import Path
import numpy as np

from ising.model.ising import IsingModel, Bias


class Netlister:

    template = Path("$TOP/ising/netlist/ising.template")
    latch = Path("$TOP/lib/components/latch/basic.scs")
    cu = Path("$TOP/lib/components/cu/resistor.scs")

    def generate(
            self,
            model: IsingModel,
            file: Path,
            initial_state: np.ndarray|None = None,
            latch_vth: np.ndarray|None = None
        ):

        if initial_state is None:
            initial_state = np.random.choice([-1, 1], size=model.num_variables)

        if latch_vth is None:
            latch_vth = np.full(model.num_variables, None, dtype=object)

        placeholders = {}
        placeholders['include_latch'] = self.latch
        placeholders['include_cu'] = self.cu
        placeholders['latch'] = '\n'.join([self.gen_latch(i, latch_vth[i]) for i in range(model.num_variables)])
        placeholders['cu'] = '\n'.join([
            self.gen_cu(i, j, model.J[i,j]) for i, j in zip(*np.triu.indices(model.num_variables))
        ])
        placeholders['ic'] = '\n'.join([self.gen_ic(i, initial_state[i]) for i in range(model.num_variables)])

        template = string.Template(self.template.read_text())
        out = template.substitue(placeholders)
        file.write_text(out)

    def gen_latch(self, i: int, vth = None) -> str:
        out = f"latch_{i} (L{i} R{i}) node"
        if vth is not None:
            out += f" vth={vth}"
        return out

    def gen_cu(self, i: int, j: int, J: Bias) -> str:
        R = 1/J * 10_000
        return f"cu_{i}_{j} (L{i} R{i} L{j} R{j}) cu R={R}"

    def gen_ic(self, i: int, ic: bool) -> str:
        if ic == 1:
            left, right = ("0", "P_supply")
        elif ic == -1:
            left, right = ("P_supply", "0")
        else:
            raise ValueError("initial_state should only contain 1 and -1 values")
        return f"ic L{i}={left} R{i}={right}"
