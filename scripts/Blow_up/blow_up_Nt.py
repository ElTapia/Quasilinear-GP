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
k = 1
N = 200_000
Nts = np.array([2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000])
tf = 2.25e-3
pars = {"k":k}

# Define total of files
Nelem = 50

# Create folder
folder = 'blow_up_A_{A}_sigma_{s}_Nts'.format(A=A, s=sigma)
create_folder(folder)
folder = folder + "/"

for j in range(len(Nts)):
    Nt = Nts[j]
    N_times = Nt/Nelem

    # Create folder
    path = folder + 'blow_up_Nt_{Nt}'.format(Nt=Nt)
    create_folder(path)
    path += "/"

    print("Solving for Nt = ", Nt, " ...")

    w_ = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path, 
                  save_as_files="w_", save_properties=True, order=2,
                 background=0)
    w_.set_init_condition(gaussian)
    w_.solve(Nelem=N_times)

print("Done!")

