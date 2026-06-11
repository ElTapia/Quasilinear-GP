import sys
sys.path.append('../../../../')

import matplotlib.pyplot as plt

# Import methods
import QGP.Defocusing.Methods as Def
import QGP.Focusing.Methods as Foc
from QGP.Soliton_Gen import *
from Utilities import *
import config

cs  = np.linspace(0, np.sqrt(2), 20, endpoint=False)
ks = np.linspace(-3, -6, 20) # k0 = -3.636

N = 800_001
L = 200
folder = "DS_DB/"
for j in range(len(cs)):
    c = cs[j]
    for l in range(len(ks)):
        k = ks[l]
        DS_name = f'DS_c_{c:.4f}_k_{k:.4f}'

        print(f'Generating soliton {j*len(ks) + l + 1} of {len(cs)*len(ks)}')
        print(f'c = {c}, k = {k}')
        soliton_, x = soliton(c, k, soliton_type="dark", L=L, N=N)

        save_file(soliton_, folder + DS_name)


