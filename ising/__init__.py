import os

num_cores = 12
os.sched_setaffinity(0, range(num_cores))
