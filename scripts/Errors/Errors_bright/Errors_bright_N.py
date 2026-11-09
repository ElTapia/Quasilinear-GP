import sys
sys.path.append('../../../src/')
import os

import pickle

# Import methods
import QGP.Focusing.Methods as Foc
from QGP.Soliton_Gen import *
from Utilities import *

i = 1j
#### Dark soliton ####

# Parameters
L = 40
c = 0
w = np.sqrt(2)/2
tf = 5
k = 1/4
pars_QGP = {"k": k}
N_theo = 262_145
order = 2

theo_name = "theo_bright_w_{w}_k_{k}_L_{L}.pck".format(w=round(w, 2), k=k, L=L)

if not os.path.exists(theo_name):
    # Theoretical soliton
    w_bright, x_bright = soliton(c, k, w, soliton_type="bright", L=L, N=N_theo, as_func=True)

    # Save initial state
    print("Saving initial function...")
    with open(theo_name, 'wb') as f:
        pickle.dump(w_bright, f)
    
    print("Initial condition saved")

else:
    w_bright = pickle.load(open(theo_name, "rb"))

def bright_theo_QGP(x, t):
    return w_bright(x) * np.exp(-i * w * t)

Ns = np.array([2**(j) for j in range(4, 18)]) #np.array([2**(j) for j in range(8, 18)])+1
Nts = np.array([20, 50, 100, 200, 500, 2000, 5000]) #np.array([100, 200, 1000, 5000])

order_folder = "space_order_{}".format(order)
create_folder(order_folder)
order_folder += "/"

for j in range(len(Nts)):
    Nt = Nts[j]
    dt = tf/Nt
    N_times = Nt/10

    # Create folder
    folder_name = order_folder + 'errors_Nt_{Nt}_order_{o}'.format(Nt=Nt, o=order)
    create_folder(folder_name)
    folder_name += "/"

    print("Nt = ", Nt, " ...")

    errors = np.zeros(len(Ns))
    save_file(errors, folder_name+"error")

    for n in range(len(Ns)):

        N = Ns[n]

        folder_N = "solution_N_{}".format(N)
        create_folder(folder_name + folder_N)
        path = folder_name + folder_N + "/"

        print("Solving for N = ", N, " ...")

        w_ = Foc.CNFD(L, N, Nt, tf, pars_QGP, "neumann", path=path, 
                      save_as_files="w_", order=order)
        w_.set_w_theo(bright_theo_QGP)
        w_.solve(Nelem=N_times)

        error = np.sqrt(w_.h) * np_norm((w_.w - bright_theo_QGP(w_.x, tf)), 2)
        errors[n] = error
        save_file(errors, folder_name+"error")

print("Done!")
