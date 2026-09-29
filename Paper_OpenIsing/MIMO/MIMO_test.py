from pathlib import Path
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

import numpy as np
import pickle
from ising import api
from ising.postprocessing.MIMO_plot import plot_error_SNR
from ising.stages import TOP
import yaml
from Paper_OpenIsing import BASE_PATH

base_path = BASE_PATH / "MIMO"
fig_path = base_path / "figures"
Path.mkdir(fig_path, parents=True, exist_ok=True)
data_path = base_path / "data"
Path.mkdir(data_path, parents=True, exist_ok=True)
config_path = base_path / "config"
Path.mkdir(config_path, parents=True, exist_ok=True)

problem_type = "MIMO"
config_file = config_path / "MIMO_test.yaml"
config_file_change = config_path / "config_MIMO_change10-5.yaml"
with config_file.open("rb") as f:
    config = yaml.safe_load(f)
snr_list = config["dummy_snr"]
solvers = config["solvers"]
nb_problems = 50

ber_results = {snr: {solver: [] for solver in solvers + ["ZF"]} for snr in snr_list}
for snr in snr_list:
    config["dummy_snr"] = snr
    for problem in range(nb_problems):
        data_file = data_path / f"{snr}_problem{problem}_ber.pkl"
        data_file_zf = data_path / f"{snr}_problem{problem}_ber_zf.pkl"
        if data_file.exists():
            with data_file.open("rb") as f:
                res = pickle.load(f)
            for solver in solvers:
                ber_results[snr][solver].append(res[solver])
            with data_file_zf.open("rb") as f:
                res = pickle.load(f)
            ber_results[snr]["ZF"].append(res)
        else:
            config["dummy_seed"] = config["initialization_seed"] + problem+40
            with config_file_change.open("w") as f:
                yaml.safe_dump(config, f)

            ans, _ = api.get_hamiltonian_energy(
                problem_type=problem_type, config_path=str(config_file_change.relative_to(TOP))
            )
            for solver in (solvers+["ZF"]):
                ber_results[snr][solver].append(ans.BER[solver])
            with data_file.open("wb") as f:
                pickle.dump(ans.BER, f, protocol=pickle.HIGHEST_PROTOCOL)
            with data_file_zf.open("wb") as f:
                pickle.dump(ans.BER["ZF"], f, protocol=pickle.HIGHEST_PROTOCOL)

    for solver in (solvers+["ZF"]):
        ber_results[snr][solver] = np.mean(ber_results[snr][solver])
for snr in snr_list:
    print(f"mean BER at SNR {snr}: {ber_results[snr]}")
plot_error_SNR(ber_results, save_folder=fig_path,fig_name="BER_curve_MIMO_test")
