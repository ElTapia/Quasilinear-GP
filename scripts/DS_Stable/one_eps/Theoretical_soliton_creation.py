import sys
sys.path.append('../../../src/')
import os
from time import time
import pickle

# Import methods
from QGP.Soliton_Gen import *
from Utilities import *
ex_ti = time()

# Init parameters
k = -50
pars = {"k":k}
c_0 = np.sqrt(2)/2

P_p_ = lambda c: P_p(c, k)
P_pp_ = lambda c: P_pp(c, k)

c_star, error = newton_1D(c_0, P_p_, P_pp_)

eps = 0.1
c_st = c_star + eps

# Values for theoretical soliton
L_theo = 20_000
N_theo = 2_000_001

theo_name = "theo_st_c_{c}_k_{k}.pck".format(c=round(c_st, 2), k=k)

if not os.path.exists(theo_name):
    # Theoretical soliton
    w_dark_st, x_dark_st = soliton(c_st, k, soliton_type="dark", L=L_theo, N=N_theo, as_func=True)

    # Save initial state
    print("Saving initial function...")
    with open(theo_name, 'wb') as f:
        pickle.dump(w_dark_st, f)
    
    print("Initial condition saved")

ex_tf = time()
execution_time = ex_tf - ex_ti
print("Execution time (including init soliton): ", execution_time)