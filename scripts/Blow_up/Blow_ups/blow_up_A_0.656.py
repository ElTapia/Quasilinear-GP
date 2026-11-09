import sys
sys.path.append('../../../src/')
import os

import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *

As = np.array([0.65625])
sigma = 1/10

def gaussian(x):
    output = A * np.exp(-x**2/(2 * sigma**2))
    return output

L = np.pi
k = 1
pars = {"k":k}

N = 160_000
Nt = 64_221
tf = 2.18e-3


# Define total of files
Nelem = 100
N_times = int(Nt/Nelem)

# Create folder
folder = 'blow_up_N_{N}_Nt_{Nt}_sigma_{s}_third'.format(N=N, Nt=Nt, s=sigma)
create_folder(folder)
folder = folder + "/"

for j in range(len(As)):
    A = As[j]

    def gaussian(x):
        output = A * np.exp(-x**2/(2 * sigma**2))
        return output

    # Create folder
    path = folder + 'blow_up_A_{A}'.format(A=round(A, 3))
    create_folder(path)
    path += "/"

    print("Solving for A = ", round(A, 3), " ...")

    w_ = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path, 
                  save_as_files="w_", save_properties=True, order=2,
                 background=0)
    w_.set_init_condition(gaussian)
    w_.solve(Nelem=N_times)

print("Done!")
