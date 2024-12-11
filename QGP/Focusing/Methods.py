import scipy as sp
from scipy.sparse import diags, identity, bmat, kron, csc_matrix
from scipy.optimize import root
from time import time

import sys
sys.path.append('../')
from Base import *
from Utilities import *


class Time_CN(NLS_numerical):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)

        self.A = self.Ah()
        self.I = identity(self.N, format="csc")


    def Ah(self):
        e = np.ones(self.N, dtype = 'float64') * -2

        if self.border.lower() == "neumann":
            e[0] = -1
            e[-1] = -1

        f = np.ones(self.N-1, dtype = 'float64')

         # Arreglo con las diagonales
        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales

        A = diags(k_, offset, format="csc", dtype='float64')

        return A


class CNFD(Time_CN):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)


    def F(self, u, v):
        k = self.pars["k"]
        w = u[:self.N]
        w_c = u[self.N:]
        i = 1j
        
        norm_w = norm(w)
        norm_v = norm(v)
        
        gp = self.dt/2 * (w + v) * (norm_w + norm_v)/2
        quasi = k * self.l/4 * (w + v) * (self.A * (norm_w + norm_v))
        F_ = i * (v - w) + self.l/2 * self.A * (w + v) + gp + quasi
        F_conj = F_.conjugate()

        output = np.append(F_, F_conj)
        return output


    def quasi_dFdw(self, w, w_c, v):
        
        norm_w = norm(w)
        norm_n = norm(v)
        
        sem_diag = w[1:] * w_c[:-1]

        # Unnecesary operations
        diag = self.A * norm_w - 2 * norm_w
        diag[0] = diag[0] + norm_w[0]
        diag[-1] = diag[-1] + norm_w[-1]

        first_term = np.array([sem_diag, diag, sem_diag.conjugate()], dtype=object) 
        offset = [-1, 0, 1]
        
        first_term = diags(first_term, offset, format="csc", dtype='complex_')


        second_term = np.array([self.A * norm_n])
        second_term = diags(second_term, [0], format="csc", dtype='complex_')


        sem_diag_1 = w_c[:-1] * v[1:]

        diag = -2 * v * w_c
        diag[0] = -v[0] * w_c[0]
        diag[-1] = -v[-1] * w_c[-1]

        sem_diag_2 = w_c[1:] * v[:-1]
        
        third_term = np.array([sem_diag_1, diag, sem_diag_2], dtype=object)
        third_term = diags(third_term, offset, format="csc", dtype='complex_')
        
        output = first_term + second_term + third_term
        return output


    def quasi_dFdw_c(self, w, w_c, v):
        
        sem_diag = w[1:] * w[:-1]
        diag = -2 * w**2
        diag[0] = -w[0]**2
        diag[-1] = -w[-1]**2

        first_term = np.array([sem_diag, diag, sem_diag], dtype=object) 
        offset = [-1, 0, 1]
        
        first_term = diags(first_term, offset, format="csc", dtype='complex_')

        sem_diag_1 = v[1:] * w[:-1]
        diag = -2 * v * w
        diag[0] = -v[0] * w[0]
        diag[-1] = -v[-1] * w[-1]
        sem_diag_2 = v[:-1] * w[1:]
        
        second_term = np.array([sem_diag_1, diag, sem_diag_2], dtype=object)
        second_term = diags(second_term, offset, format="csc", dtype='complex_')
        
        output = first_term + second_term
        return output


    def F_prime(self, u, v):
        k = self.pars["k"]
        w = u[:self.N]
        w_c = u[self.N:]

        i = 1j

        norm_w = norm(w)
        norm_n = norm(v)


         # Arreglo con las diagonales
        k_dfdw = np.array([self.dt/4 * (2 * norm_w + v * w_c + norm_n)]) 
        k_dfdw_c = np.array([self.dt/4 *(w**2 + v * w)]) 
        offset = [0]

        dFdw_nl = diags(k_dfdw, offset, format="csc", dtype='complex_')
        dFdw_c_nl = diags(k_dfdw, offset, format="csc", dtype='complex_')
        
        if k==0:
            dFdw = (-i + self.dt/2) * self.I + self.l/2 * self.A - dFdw_nl
            dFdw_c = -dFdw_c_nl
            dF_cdw = dFdw_c.conjugate()
            dF_cdw_c = dFdw.conjugate()

            jac = bmat([[dFdw, dFdw_c], [dF_cdw, dF_cdw_c]], format="csc", dtype="complex_")
            return jac

        quasi_dFdw = self.quasi_dFdw(w, w_c, v)
        quasi_dFdw_c = self.quasi_dFdw_c(w, w_c, v)
        
        dFdw = (-i + self.dt/2) * self.I + self.l/2 * self.A - dFdw_nl + k * self.l/4 * quasi_dFdw
        dFdw_c = -dFdw_c_nl + k * self.l/4 * quasi_dFdw_c
        dF_cdw = dFdw_c.conjugate()
        dF_cdw_c = dFdw.conjugate()

        jac = bmat([[dFdw, dFdw_c], [dF_cdw, dF_cdw_c]], format="csc", dtype="complex_")
        return jac


    def update(self, n):
        i=1j
        # Actual state
        wn = self.w[n]
        wn_c = wn.conjugate()
        wn_to_solve = np.append(wn, wn_c)

        def F_(X):
            output = self.F(X, wn)
            return output

        def F_prime_(X):
            output = self.F_prime(X, wn)
            return output

        w_n_1, error = newton(wn_to_solve, F_, F_prime_)

        self.w[n+1] = w_n_1[:self.N]