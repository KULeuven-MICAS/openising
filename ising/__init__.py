import os

num_cores = 8
os.sched_setaffinity(0, range(num_cores))
