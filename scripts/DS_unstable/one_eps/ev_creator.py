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
c_uns = c_star - eps

L = 10_000
N = 1_000_000


# Create folder
folder_name = 'Evs'
create_folder(folder_name)
path = folder_name + "/"


# Load theoretical
theo_name = "theo_uns_c_{c}_k_{k}.pck".format(c=round(c_uns, 2), k=k)
if not os.path.exists(theo_name):
    print("Theoretical solution doesn't exist!")

else:
    w_dark_uns = pickle.load(open(theo_name, "rb"))


# Get eigenvalues and eigenvectors
x = np.linspace(-L, L, N)
h = x[1] - x[0]
us = w_dark_uns(x)

D1 = D1_neumann_centered(N, h)
D2 = D2_neumann_centered(N, h)

M, A, B = build_linearized_matrix(us, D1, D2, k, c_uns)

print("Obtaining eigenvalues and eigenvectors...")
evals, evecs = spla.eigs(M, k=1, sigma=0.1, which='LR')
print("Eigenvalue: ", evals)

print("Saving values...")
save_file(evals, path+"evals")
save_file(evecs, path+"evecs")

print("Done !")
ex_tf = time()
execution_time = ex_tf - ex_ti
print("Execution time: ", execution_time)
