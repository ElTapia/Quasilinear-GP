import sys
sys.path.append('../../../src/')
import os
from time import time
import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *
from Eigenvalues import *

ex_ti = time()

# Init parameters
k = -50
pars = {"k":k}
c_0 = np.sqrt(2)/2

P_p_ = lambda c: P_p(c, k)
P_pp_ = lambda c: P_pp(c, k)

c_star, error = newton_1D(c_0, P_p_, P_pp_)

eps = 0.1
c_uns = c_star - 3 * eps

L = 10_000
N = 5_000_000

tf = 8
Nt = 200_000 #10_000

# Define total of files
Nelem = 200
N_times = int(Nt/Nelem)


# Create folder
eps_str = str(eps).replace('.', '')
folder_name = 'uns_k_{k}_ev_N_{N}_Nt_{Nt}_L_{L}'.format(k=k, N=N, Nt=Nt, L=L)
create_folder(folder_name)
path = folder_name + "/"

# Load theoretical
theo_name = "theo_uns_c_{c}_k_{k}.pck".format(c=round(c_uns, 2), k=k)
if not os.path.exists(theo_name):
    print("Theoretical solution doesn't exist!")

else:
    w_dark_uns = pickle.load(open(theo_name, "rb"))


# Eigenvalues and eigenvectors path
evals_path = "Evs/evals.npy.gz"
evecs_path = "Evs/evecs.npy.gz"


# Load eigenvalues
if not os.path.exists(evals_path):
    print("Eigenvalues don't exist!")

else:
    evals_file = gzip.GzipFile(evals_path, "r")
    evals = np.load(evals_file)


# Load eigenvectors
if not os.path.exists(evecs_path):
    print("Eigenvalues don't exist!")

else:
    evecs_file = gzip.GzipFile(evecs_path, "r")
    evecs = np.load(evecs_file)


# Get eta perturbation
eta = get_eta(evals, evecs, N)

# Init condition
def init_cond_pert(x, w_uns, eta, eps=1e-2):
    output = w_uns(x) + eps * eta
    return output

init_cond = lambda x: init_cond_pert(x, w_dark_uns, eta)

# Set theo solution
def w_dark_theo(x, t):
    output = w_dark_uns(x - c_uns * t)
    return output

# Simulation
print("Solving...")
w_un = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path,
                save_as_files="w_", save_properties=True, order=2)

w_un.w_theo = w_dark_theo
w_un.set_init_condition(init_cond)
w_un.solve(Nelem=N_times, b_tol=1e-2, energy_perc=.1)


print("Done!")
ex_tf = time()
execution_time = ex_tf - ex_ti
print("Execution time (including init soliton): ", execution_time)
