import sys
sys.path.append('../../../src/')
import os

import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *

i = 1j
#### Dark soliton ####

# Parameters
L = 100
c = np.sqrt(2)/2
tf = 5
k = 1/4
pars_QGP = {"k": k}
N_theo = 262_145
order = 2

theo_name = "theo_c_{c}_k_{k}_L_{L}.pck".format(c=round(c, 2), k=k, L=L)

if not os.path.exists(theo_name):
    # Theoretical soliton
    w_dark, x_dark = soliton(c, k, soliton_type="dark", L=L, N=N_theo, as_func=True)

    # Save initial state
    print("Saving initial function...")
    with open(theo_name, 'wb') as f:
        pickle.dump(w_dark, f)
    
    print("Initial condition saved")

else:
    w_dark = pickle.load(open(theo_name, "rb"))

def dark_theo_QGP(x, t):
    return w_dark(x - c * t)

# Matches the values actually plotted in Dark_D1_N_L_100_c_0.70_k_0.25.pdf
Ns = np.array([2**p for p in range(7, 18)])
Nts = np.array([10, 50, 100, 200, 500, 2000])

order_folder = "space_order_2_L_{}".format(L)
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

        w_ = Def.CNFD(L, N, Nt, tf, pars_QGP, "neumann", path=path,
                      save_as_files="w_", order=order)
        w_.set_w_theo(dark_theo_QGP)
        w_.solve(Nelem=N_times)

        diff = w_.w - dark_theo_QGP(w_.x, tf)

        error = np.sqrt(w_.h) * np_norm(diff, 2)
        errors[n] = error
        save_file(errors, folder_name+"error")

        error_H1 = w_.norm_H1_dx(diff)
        errors_H1[n] = error_H1
        save_file(errors_H1, folder_name+"error_H1")

print("Done!")

