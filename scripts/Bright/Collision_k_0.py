import sys
sys.path.append('../../src/')
import os

import pickle

# Import methods
import QGP.Focusing.Methods as Foc
from QGP.Soliton_Gen import *
from Utilities import *

i = 1j

L_stand = 600
N_stand = 800_001

L = 400
N = 500_000
Nt = 80_000

k = 0
w = 1/2
x_0 = 20
tf = 100

pars = {"k":k}
c = np.sqrt(2)/2
c_1 = c
c_2 = -c

# Define total of files
Nelem = 100
N_times = int(Nt/Nelem)
 
# Create folder
folder_name = 'bright_collision_N_{N}_Nt_{Nt}_L_{L}_w_{w}_k_{k}_c_{c}'.format(N=N, Nt=Nt, L=L, w=w, k=k, c=round(c, 2))
create_folder(folder_name)
path = folder_name + "/"

theo_stationary = "theo_bright_stat_k_{k}_w_{w}_L_{L}.pck".format(k=k, w=w, L=L_stand)

if not os.path.exists(path+theo_stationary):
    # Theoretical soliton
    w_bright, x_bright = soliton(0, k, w=w, soliton_type="bright", L=L_stand, N=N_stand, as_func=True)
    
    # Save initial state
    #print("Saving initial function...")
    #with open(path+theo_stationary, 'wb') as f:
    #    pickle.dump(w_bright, f)
    
    #print("Initial condition saved")

else:
    w_bright = pickle.load(open(path+theo_stationary, "rb"))


def init_collision(x, t=0):
    i = 1j
    left_soliton = w_bright(x-c_1*t+x_0) * np.exp(-i*(c_1**2/4*t+c_1/2*x+w*t))
    right_soliton = w_bright(x-c_2*t-x_0) * np.exp(-i*(c_2**2/4*t+c_2/2*x+w*t))

    output = left_soliton + right_soliton
    return output

# Get L_tilde
x_col = np.linspace(-L, L, N)
x_pos = x_col[x_col>0]
sol_norm = norm(w_bright(x_pos))

zero_ind = np.where(np.isclose(sol_norm, 0, atol=1e-6))[0][0]
x_zero = x_pos[zero_ind]

L_tilde = np.abs(x_zero)

# Simulation
print("Solving...")
w_sim = Foc.CNFD(L, N, Nt, tf, pars, "neumann", x_0=x_0, L_tilde=L_tilde, path=path,
                save_as_files="w_", save_properties=True, order=2, local=True)

w_sim.set_init_condition(init_collision)
w_sim.solve(Nelem=N_times, b_tol=1e-5)

print("L_tilde used: ", L_tilde)
print("Done!")
