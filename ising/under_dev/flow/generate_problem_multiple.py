import problem_generation
from sys import argv
import numpy as np

parser = argv.argparse.ArgumentParser()
parser.add_argument('-nb', help='Number of problems to generate', default=1)
parser.add_argument('-file', help='filename where everything is stored', default='Gen_problems')
parser.add_argument('-Nmin', help='Minimum size of the problem', default=3)
parser.add_Argument('-Nmax', help='Maximum size of the problem', default=100)
parser.add_Argument('-Jmin', help='Minimum value for activation weights', default=-1.)
parser.add_Argument('-Jmax', help='Maximum value for activation weights', default=1.)
parser.add_Argument('-hmin', help='Minimum value for magnetic field weights', default=-1.)
parser.add_Argument('-hmax', help='Maximum value for magnetic field weights', default=1.)

args = parser.parse_args()
nb = int(args.nb)
Nmin = int(args.Nmin)
Nmax = int(args.Nmax)
step = int((Nmax - Nmin)/nb)
Jmin = float(args.Jmin)
Jmax = float(args.Jmax)
hmin = float(args.hmin)
hmax = float(args.hmax)
file = args.file

print("Starting generating problem and storing them in " + file)

for i in range(Nmin, Nmax, step):
    J, h, optim = problem_generation.problem_gen(i, Jmin, Jmax, hmin, hmax)
    fileName = f'{file}_{i}'
    np.savez(fileName, J=J, h=h, optim_state=optim[0], optim_value=optim[1])


print("Completed generating problems")