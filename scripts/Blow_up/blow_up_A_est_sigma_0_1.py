import sys
sys.path.append('../../src/')
import os

import pickle

# Import methods
import QGP.Defocusing.Methods as Def
from QGP.Soliton_Gen import *
from Utilities import *

sigma = 1/10

L = np.pi
k = 1
pars = {"k":k}

N = 100_000
Nt = 5_000 #32_150 #64_300
T = 4e-3

# Define total of files
N_times = 0.1 # Prevent to save all the solutions

folder = 'blow_up_A_est_sigma_{s}'.format(s=sigma)
create_folder(folder)
path = folder + "/"

sols_folder = path + "solutions"
create_folder(sols_folder)
sols_path = sols_folder + "/"

def A_bl_guess(Nt, M, A_guess):

    def gaussian(x, A = A_guess):
        output = A * np.exp(-x**2/(2 * sigma**2))
        return output

    
    w_Nt = Def.CNFD(L, N, Nt, T, pars, "neumann", path=sols_path, 
              save_as_files="w_{A}_".format(A=round(A_guess, 5)), order=2, background=0, print_iteration=False)
    w_Nt.set_init_condition(gaussian)
    
    w_Nt_double = Def.CNFD(L, N, 2*Nt, T, pars, "neumann", path=sols_path, 
              save_as_files="w_{A}_2Nt".format(A=round(A_guess, 5)), order=2, background=0, print_iteration=False)
    w_Nt_double.set_init_condition(gaussian)

    w_max = w_Nt.solve(Nelem=N_times, return_max=True)
    w_max_double = w_Nt_double.solve(Nelem=N_times, return_max=True)

    w_max_double = w_max_double[::2]

    w_max[w_max==0] = np.inf
    w_max_double[w_max_double==0] = np.inf

    rate = np.max(np.abs(w_max_double/w_max))
    
    return w_max, w_max_double, rate < M


A_guesses = np.zeros(100)


def bisection(A_g_l, A_g_r, Nt, M, tolerance=1e-8, max_iter=1000):
    j = 1
    guess = False
    A_g = (A_g_r + A_g_l)/2
    A_guesses[0] = A_g

    while np.abs(A_g_l - A_g_r) > tolerance and j < max_iter:

        A_g = (A_g_r + A_g_l) / 2

        print("Actual A = ", A_g)
        w_max, w_max_double, guess = A_bl_guess(Nt, M, A_g)

        if guess:
            A_g_l = A_g

        else:
            A_g_r = A_g

        A_guesses[j] = A_g
        save_file(A_guesses, path+"A_guesses")
        j += 1


save_file(A_guesses, path+"A_guesses")

M = 1.5
A_g_l = 0.631
A_g_r = 0.638
bisection(A_g_l, A_g_r, Nt, M, max_iter=1000)

print("Done!")