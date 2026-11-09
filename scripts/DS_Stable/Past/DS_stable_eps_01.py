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
c_st = c_star + eps

L = 200
N = 250_000 #500_001
tf = 15
Nt = 15_000 #tf/dt

# Define total of files
Nelem = 100
N_times = Nt/Nelem

# Create folder
eps_str = str(eps).replace('.', '')
folder_name = 'st_k_{k}_eps_{e}_N_{N}_Nt_{Nt}_L_{L}'.format(k=k, e=eps_str, N=N, Nt=Nt, L=L)
create_folder(folder_name)
path = folder_name + "/"

theo_name = "theo_c_{c}_k_{k}_L_{L}.pck".format(c=round(c_st, 2), k=k, L=L)

if not os.path.exists(path+theo_name):
    # Theoretical soliton
    w_dark_st, x_dark_st = soliton(c_st, k, soliton_type="dark", L=L, N=N+1, as_func=True)
    
    # Save initial state
    print("Saving initial function...")
    with open(path+theo_name, 'wb') as f:
        pickle.dump(w_dark_st, f)
    
    print("Initial condition saved")

else:
    w_dark_st = pickle.load(open(path+theo_name, "rb"))


def w_dark_theo(x, t):
    output = w_dark_st(x - c_st * t)
    return output

# Simulation
print("Solving...")
w_st = Def.CNFD(L, N, Nt, tf, pars, "neumann", path=path,
                save_as_files="w_", save_properties=True, order=2)
w_st.set_w_theo(w_dark_theo)
w_st.solve(Nelem=N_times)

print("Done!")