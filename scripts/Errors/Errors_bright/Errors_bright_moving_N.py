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
L = 100
c = np.sqrt(2)/2
w = 1
tf = 5
k = 1/4
x_0 = 0
pars_QGP = {"k": k}

L_theo = 200
N_theo = 262_145
order = 2

theo_name = "theo_bright_c_{c}_k_{k}_w_{w}_L_{L}.pck".format(c=round(c, 2), k=k, w=w, L=L)

if not os.path.exists(theo_name):
    # Theoretical soliton
    w_bright, x_bright = soliton(c, k, w=w, soliton_type="bright", L=L_theo, N=N_theo, as_func=True)

    # Save initial state
    print("Saving initial function...")
    with open(theo_name, 'wb') as f:
        pickle.dump(w_bright, f)
    
    print("Initial condition saved")

else:
    w_bright = pickle.load(open(theo_name, "rb"))

def bright_theo_QGP(x, t, c=c, w=w, x_0=x_0):
    i = 1j
    output = w_bright(x-c*t-x_0) * np.exp(-i*(c**2/4*t+w*t))
    return output

# Matches the values actually plotted in Bright_N_L_100_c_0.71_k_0.25.pdf
Ns = np.array([2**p for p in range(7, 18)])
Nts = np.array([50, 100, 200, 500, 2000, 5000])

order_folder = "moving_space_order_{}".format(order)
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
    errors_H1 = np.zeros(len(Ns))
    save_file(errors, folder_name+"error")
    save_file(errors_H1, folder_name+"error_H1")

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

        diff = w_.w - bright_theo_QGP(w_.x, tf)

        error = np.sqrt(w_.h) * np_norm(diff, 2)
        errors[n] = error
        save_file(errors, folder_name+"error")

        error_H1 = w_.norm_H1_dx(diff)
        errors_H1[n] = error_H1
        save_file(errors_H1, folder_name+"error_H1")

print("Done!")
