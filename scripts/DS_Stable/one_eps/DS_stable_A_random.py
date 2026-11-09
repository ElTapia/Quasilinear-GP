import sys
sys.path.append('../../../src/')
import os
from time import time
import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *
ex_ti = time()

# Set random seed
np.random.seed(67)

# Init parameters
k = -50
pars = {"k":k}
c_0 = np.sqrt(2)/2

P_p_ = lambda c: P_p(c, k)
P_pp_ = lambda c: P_pp(c, k)

c_star, error = newton_1D(c_0, P_p_, P_pp_)

eps = 0.1
c_st = c_star + eps

L = 10_000
N = 1_000_000

tf = 10 #60 #30
Nt = 100_000 #10_000

# Define total of files
Nelem = 200

# Create folder
eps_str = str(eps).replace('.', '')
folder_name = 'st_k_{k}_rand_N_{N}_Nt_{Nt}_L_{L}'.format(k=k, N=N, Nt=Nt, L=L)
create_folder(folder_name)
path = folder_name + "/"

# Load theoretical
theo_name = "theo_st_c_{c}_k_{k}.pck".format(c=round(c_st, 2), k=k)
if not os.path.exists(theo_name):
    print("Theoretical solution doesn't exist!")

else:
    w_dark_st = pickle.load(open(theo_name, "rb"))


def init_gauss(x):
    eps_rand = 1e-2
    
    A_rand = np.random.rand(len(x)) * eps_rand
    A_rand[np.abs(x)>=L/2] = eps_rand
    
    output = w_dark_st(x) + A_rand * np.exp(-x**2)
    return output

def w_dark_theo(x, t):
    output = w_dark_st(x - c_st * t)
    return output

# Simulation
print("Solving...")
w_un = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path,
                save_as_files="w_", save_properties=True, order=2)

w_un.w_theo = w_dark_theo
w_un.set_init_condition(init_gauss)
w_un.solve(Nelem=N_times, b_tol=1e-2, energy_perc=.1)

print("Done!")
ex_tf = time()
execution_time = ex_tf - ex_ti
print("Execution time (including init soliton): ", execution_time)
