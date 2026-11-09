import sys
sys.path.append('../../../src/')
import os

import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *


# Init parameters
k = -50
pars = {"k":k}
c_0 = np.sqrt(2)/2

P_p_ = lambda c: P_p(c, k)
P_pp_ = lambda c: P_pp(c, k)

c_star, error = newton_1D(c_0, P_p_, P_pp_)

eps = 0.1
c_in = c_star - eps

def G_c_k(c, k):

    d2eta_zero = (c**2 - c**4 - 1)/(1 - k * c**2)
    d2rho_zero = -(np.sqrt(2) / (2 * c)) * d2eta_zero
    const_1 = - (np.sqrt(2) * c * ((2 - c**2) / (2 * c))**2 + 2 * np.sqrt(2) * c - 2 * d2rho_zero)
    const_2 = d2rho_zero * np.sqrt(2) * c

    coeffs = [-4, const_1, const_2]
    roots = np.roots(coeffs)
    return roots

roots = G_c_k(c_in, k)

L = 800
N = 500_000 #500_001
tf = 45 #60 #30
Nt = 100_000 # 50_000 #tf/dt

# Define total of files
Nelem = 200
N_times = int(Nt/Nelem)
 
# Create folder
eps_str = str(eps).replace('.', '')
folder_name = 'uns_k_{k}_gauss_007_N_{N}_Nt_{Nt}_L_{L}'.format(k=k, N=N, Nt=Nt, L=L)
create_folder(folder_name)
path = folder_name + "/"

theo_name = "theo_c_{c}_k_{k}_L_{L}.pck".format(c=round(c_in, 2), k=k, L=L)

if not os.path.exists(path+theo_name):
    # Theoretical soliton
    w_dark_in, x_dark_in = soliton(c_in, k, soliton_type="dark", L=L, N=N+1, as_func=True)
    
    # Save initial state
    print("Saving initial function...")
    with open(path+theo_name, 'wb') as f:
        pickle.dump(w_dark_in, f)
    
    print("Initial condition saved")

else:
    w_dark_in = pickle.load(open(path+theo_name, "rb"))

def init_gauss(x):
    A = roots[1] - 0.007 #1e-1
    output = w_dark_in(x) + A*np.exp(-x**2)
    return output

def w_dark_theo(x, t):
    output = w_dark_in(x - c_in * t)
    return output

# Simulation
print("Solving...")
w_in = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path,
                save_as_files="w_", save_properties=True,order=2)

w_in.w_theo = w_dark_theo
w_in.set_init_condition(init_gauss)
w_in.solve(Nelem=N_times, b_tol=1e-4)

print("Done!")
ex_tf = time()
execution_time = ex_tf - ex_ti
print("Execution time (including init soliton): ", execution_time)