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
L = 40
c = np.sqrt(2)/2
tf = 5
k = 1/4
pars_QGP = {"k": k}
N_theo = 262_145
order = 2

theo_name = "theo_c_{c}_k_{k}.pck".format(c=round(c, 2), k=k)

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

Ns = np.array([100_000]) #500, 1_000, 2_000, 5_000, 10_000, 50_000,  

dts = np.array([2**(-j) for j in range(1, 14)]) # 1, 11
Nts = tf/dts

order_folder = "time_order_2"
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
    save_file(errors, folder_name+"error")

    for j in range(len(Nts)):

        Nt = int(Nts[j])
        dt = dts[j]
        N_times = Nt/10

        folder_N = "solution_Nt_{}".format(Nt)
        create_folder(folder_name + folder_N)
        path = folder_name + folder_N + "/"

        print("Solving for Nt = ", Nt, " ...")

        w_ = Def.CNFD(L, N, Nt, tf, pars_QGP, "neumann", path=path, 
                      save_as_files="w_", order=order)
        w_.set_w_theo(dark_theo_QGP)
        w_.solve(Nelem=N_times)

        error = np.sqrt(w_.h) * np_norm((w_.w - dark_theo_QGP(w_.x, tf)), 2)
        errors[j] = error
        save_file(errors, folder_name+"error")

print("Done!")
