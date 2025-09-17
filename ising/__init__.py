import os

cores_nb = int(os.getenv("AMOUNT_CORES"))
os.sched_setaffinity(0, range(cores_nb))
