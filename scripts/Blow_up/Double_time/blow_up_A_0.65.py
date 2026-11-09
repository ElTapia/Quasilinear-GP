import sys
sys.path.append('../../../src/')
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
pars = {"k":k}

N = 160_000 #200_000
Nt = 2*64_221 #64_300
tf = 2.16e-3

# Define total of files
N_elem = 100 #150
N_times = int(Nt/N_elem) #my_lin(0, Nt, N_elem, 0.5)[1:]

#first_N_times = my_lin(0, 2*64_200, 100, 0.5)[1:-1]
#second_N_times = np.arange(2*64_200, 2*64_256)
#N_times = np.append(first_N_times, second_N_times)

# Create folder
#folder = 'blow_up_N_{N}_Nt_{Nt}_sigma_{s}_A_{A}'.format(N=N, Nt=Nt, s=sigma, A=round(A, 3))
#create_folder(folder)
#path = folder + "/"

folder = 'blow_up_N_{N}_Nt_{Nt}_sigma_{s}_A_{A}'.format(N=N, Nt=Nt, s=sigma, A=round(A, 3))
create_folder(folder)
path = folder + "/"

def gaussian(x):
    output = A * np.exp(-x**2/(2 * sigma**2))
    return output

print("Solving for A = ", round(A, 3), " ...")

w_ = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path, 
              save_as_files="w_", save_properties=True, order=2,
              background=0)
w_.set_init_condition(gaussian)
w_.solve(Nelem=N_times)

print("Done!")
