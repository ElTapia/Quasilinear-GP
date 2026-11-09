import sys
sys.path.append('../../src/')
import os

import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *

A = 0.65
sigma = 1/10

def gaussian(x):
    output = A * np.exp(-x**2/(2 * sigma**2))
    return output

L = np.pi
Nt = 32_000
k = 1
Ns = np.array([5_000, 10_000, 20_000, 40_000, 80_000, 160_000])
tf = 2.25e-3
pars = {"k":k}

# Define total of files
Nelem = 50
N_times = Nt/Nelem

# Create folder
folder = 'blow_up_A_{A}_sigma_{s}_Ns'.format(A=A, s=sigma)
create_folder(folder)
folder = folder + "/"

for j in range(len(Ns)):
    N = Ns[j]
    # Create folder
    path = folder + 'blow_up_N_{N}'.format(N=N)
    create_folder(path)
    path += "/"

    print("Solving for N = ", N, " ...")

    w_ = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path, 
                  save_as_files="w_", save_properties=True, order=2,
                 background=0)
    w_.set_init_condition(gaussian)
    w_.solve(Nelem=N_times)

print("Done!")
