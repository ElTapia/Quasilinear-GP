import sys
sys.path.append('../../../src/')
import os

import pickle

# Import methods
import QGP.Focusing.Methods as Foc
from QGP.Soliton_Gen import *
from Utilities import *

i = 1j
#### bright soliton ####

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

# Matches the values actually plotted in Bright_NT_L_100_c_0.71_k_0.25_w_1.pdf
Ns = np.array([5_000, 10_000, 50_000, 100_000])

dts = np.array([2**(-j) for j in range(2, 14)])
Nts = tf/dts

order_folder = "moving_time_order_{}".format(order)
create_folder(order_folder)
order_folder += "/"

for n in range(len(Ns)):
    N = Ns[n]

    # Create folder
    folder_name = order_folder + 'errors_N_{N}_order_{o}'.format(N=N, o=order)
    create_folder(folder_name)
    folder_name += "/"

    print("N = ", N, " ...")

    errors = np.zeros(len(Nts))
    errors_H1 = np.zeros(len(Nts))
    save_file(errors, folder_name+"error")
    save_file(errors_H1, folder_name+"error_H1")

    for j in range(len(Nts)):

        Nt = int(Nts[j])
        dt = dts[j]
        N_times = Nt/10

        folder_N = "solution_Nt_{}".format(Nt)
        create_folder(folder_name + folder_N)
        path = folder_name + folder_N + "/"

        print("Solving for Nt = ", Nt, " ...")

        w_ = Foc.CNFD(L, N, Nt, tf, pars_QGP, "neumann", path=path,
                      save_as_files="w_", order=order)
        w_.set_w_theo(bright_theo_QGP)
        w_.solve(Nelem=N_times)

        diff = w_.w - bright_theo_QGP(w_.x, tf)

        error = np.sqrt(w_.h) * np_norm(diff, 2)
        errors[j] = error
        save_file(errors, folder_name+"error")

        error_H1 = w_.norm_H1_dx(diff)
        errors_H1[j] = error_H1
        save_file(errors_H1, folder_name+"error_H1")

print("Done!")
