import scipy as sp
from scipy.sparse import diags, identity, bmat, kron, csc_matrix
from scipy.optimize import root
from time import time

import sys
sys.path.append('../../')
from Base import *
from Utilities import *

class Focusing_QGP(NLS_numerical):

    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):

        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.l = self.dt/(self.h**2)
        self.order = order

    def get_mass(self, plot=False, name=None):

        if self.save_as_files is not None and self.save_properties:
            if self.local:
                wn_r = self.w[np.abs(self.x-self.x_0)<self.L_tilde]
                m_r = self.h * np.sum(norm(wn_r))
    
                self.to_save_mass_R[self.save_iter] = m_r

                wn_l = self.w[np.abs(self.x+self.x_0)<self.L_tilde]
                m_l = self.h * np.sum(norm(wn_l))
    
                self.to_save_mass_L[self.save_iter] = m_l

            wn = self.w
            m = self.h * np.sum(norm(wn))
    
            self.to_save_mass[self.save_iter] = m

        else:
            mass = np.zeros(self.Nt)
            for n in range(self.Nt):
                m = self.h * np.sum(norm(self.w[n]))
                mass[n] = m
                
            if plot:
                plt.plot(self.t, mass-mass[0], ".")
    
                plt.xlabel(r"$t$")
                plt.ylabel(r"$N(t)-N(0)$")
                plt.tight_layout()
    
                if name is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
    
            return mass


    def get_energy(self, plot=False, name=None):
        k = self.pars["k"]

        if self.save_as_files is not None and self.save_properties:
            if self.local:
                wn_r = self.w[np.abs(self.x-self.x_0)<self.L_tilde]

                e_r = self.h * np.sum(1/2 * norm(np.gradient(wn_r, self.h)) - 
                                1/4 * norm(wn_r)**2 + 
                                k/4 * (np.gradient(norm(wn_r), self.h))**2)

                wn_l = self.w[np.abs(self.x+self.x_0)<self.L_tilde]

                e_l = self.h * np.sum(1/2 * norm(np.gradient(wn_l, self.h)) - 
                                1/4 * norm(wn_l)**2 + 
                                k/4 * (np.gradient(norm(wn_l), self.h))**2)

                self.to_save_energy_R[self.save_iter] = e_r
                self.to_save_energy_L[self.save_iter] = e_l

            wn = self.w

            e = self.h * np.sum(1/2 * norm(np.gradient(wn, self.h)) - 
                                    1/4 * norm(wn)**2 + 
                                    k/4 * (np.gradient(norm(wn), self.h))**2)

            self.to_save_energy[self.save_iter] = e

        else:
            energy = np.zeros(self.Nt)
            for n in range(self.Nt):
                wn = self.w[n]
                e = self.h * np.sum(1/2 * norm(np.gradient(wn, self.h)) - 
                                1/4 * norm(wn)**2 + 
                                k/4 * (np.gradient(norm(wn), self.h))**2)
                energy[n] = e
        
            if plot:
                energy = np.trim_zeros(energy)
                plt.plot(self.t, energy-energy[0], ".")
        
                plt.xlabel(r"$t$")
                plt.ylabel(r"$E(t)-E(0)$")
                plt.tight_layout()
        
                if name is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
        
            return energy

class Time_CN(Focusing_QGP):

    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):

        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, order, local, print_iteration)
        
        self.l = self.dt/(self.h**2)
        self.A = self.Ah()
        self.I = identity(self.N, format="csc")


    def Ah(self):
        e = np.ones(self.N, dtype = 'float64') * -2

        if self.border.lower() == "neumann" and self.order==1:
            e[0] = -1
            e[-1] = -1

        f = np.ones(self.N-1, dtype = 'float64')

        if self.border.lower() == "neumann" and self.order==2:
            f[0] = 2
            # Arreglo con las diagonales
            k_ = np.array([np.flip(f),e,f], dtype=object) 
            offset = [-1,0,1] #Posiciones de las diagonales

            A = diags(k_, offset, format="csc", dtype='float64')

            return A

         # Arreglo con las diagonales
        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales

        A = diags(k_, offset, format="csc", dtype='float64')

        return A


class CNFD(Time_CN):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, order, local, print_iteration)


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
        
        norm_w = w * w_c #norm(w)
        norm_n = norm(v)
        
        sem_diag_1 = (w[1:] + v[1:]) * w_c[:-1]            

        e = np.ones(self.N, dtype = 'float64') * -4

        if self.border.lower() == "neumann":
            e[0] = -2
            e[-1] = -2

        f = np.ones(self.N-1, dtype = 'float64')

         # Arreglo con las diagonales
        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales

        B = diags(k_, offset, format="csc", dtype='float64')

        B_w = B * norm_w
        A_n = self.A * norm_n

        diag = B_w - 2 * v * w_c + A_n
        diag[0] = B_w[0] - v[0] * w_c[0] + A_n[0]
        diag[-1] = B_w[-1] - v[-1] * w_c[-1] + A_n[-1]

        sem_diag_2 = (w[:-1] + v[:-1]) * w_c[1:]

        if self.order==2:
            sem_diag_1[-1] = 2 * sem_diag_1[-1]

            diag[0] = 2*(B_w[0] - v[0] * w_c[0]) + A_n[0]
            diag[-1] = 2*(B_w[-1] - v[-1] * w_c[-1]) + A_n[0]

            sem_diag_2[0] = 2 * sem_diag_2[0]


        k_ = np.array([sem_diag_1, diag, sem_diag_2], dtype=object) 
        output = diags(k_, offset, format="csc", dtype='complex')

        return output


    def quasi_dFdw_c(self, w, w_c, v):
        
        sem_diag_1 = w[1:] * w[:-1] + v[1:] * w[:-1]

        diag = -2 * w**2 - 2 * v * w
        diag[0] = -w[0]**2 - v[0] * w[0]
        diag[-1] = -w[-1]**2 - v[-1] * w[-1]

        sem_diag_2 = w[1:] * w[:-1] + v[:-1] * w[1:]

        if self.order==2:
            sem_diag_1[-1] = 2 * sem_diag_1[-1]

            diag[0] = 2 * diag[0]
            diag[-1] = 2 * diag[-1]

            sem_diag_2[0] = 2 * sem_diag_2[0]

        k_ = np.array([sem_diag_1, diag, sem_diag_2], dtype=object) 
        offset = [-1, 0, 1]

        output = diags(k_, offset, format="csc", dtype='complex')
        
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

        dFdw_nl = diags(k_dfdw, offset, format="csc", dtype='complex')
        dFdw_c_nl = diags(k_dfdw, offset, format="csc", dtype='complex')
        
        if k==0:
            dFdw = -i * self.I + self.l/2 * self.A + dFdw_nl
            dFdw_c = dFdw_c_nl
            dF_cdw = dFdw_c.conjugate()
            dF_cdw_c = dFdw.conjugate()

            jac = bmat([[dFdw, dFdw_c], [dF_cdw, dF_cdw_c]], format="csc", dtype="complex")
            return jac

        quasi_dFdw = self.quasi_dFdw(w, w_c, v)
        quasi_dFdw_c = self.quasi_dFdw_c(w, w_c, v)
        
        dFdw = -i * self.I + self.l/2 * self.A + dFdw_nl + k * self.l/4 * quasi_dFdw
        dFdw_c = dFdw_c_nl + k * self.l/4 * quasi_dFdw_c
        dF_cdw = dFdw_c.conjugate()
        dF_cdw_c = dFdw.conjugate()

        jac = bmat([[dFdw, dFdw_c], [dF_cdw, dF_cdw_c]], format="csc", dtype="complex")
        return jac


    def G(self, u, v):

        i=1j
        I = identity(self.N)
        k = self.pars["k"]
        
        norm_u = norm(u)
        norm_v = norm(v)

        linear = i * v + self.l/2 * self.A * v
        gp = self.dt/4 * (u + v) * (norm_u + norm_v)
        quasi = k * self.l/4 * (u + v) * (self.A * (norm_u + norm_v))

        non_linear = linear + gp + quasi
        
        M_to_invert = i * I - self.l/2 * self.A
        output = spsolve(M_to_invert, non_linear)
        
        return output


    def update(self, n, **kwargs):
        i=1j
        method = "newton"

        if kwargs:
            method = kwargs["method"]

        if self.save_as_files is None:
            # Actual state
            wn = self.w[n]

        else:
            wn = self.w

        wn_c = wn.conjugate()
        wn_to_solve = np.append(wn, wn_c)

        def F_(X):
            output = self.F(X, wn)
            return output

        def F_prime_(X):
            output = self.F_prime(X, wn)
            return output

        def G_(X):
            output = self.G(X, wn)
            return output

        if method.lower()=="newton":
            w_n_1, error, iteration = newton(wn_to_solve, F_, F_prime_, self.h)
            if self.print_iteration:
                print("Iteration ",n+1, " of ", self.Nt, ". Error: ", error, ". Newton iterations: ", iteration)

            if error != error:
                self.stop = True
                pass

            return w_n_1[:self.N]


        elif method.lower()=="fixed point":
            w_n_1, error, iteration = fixed_point(wn, G_, self.h)
            if self.print_iteration:
                print("Iteration ",n+1, " of ", self.Nt, ". Error: ", error, ". Fixed iterations: ", iteration)

            if error != error:
                self.stop = True
                pass

            return w_n_1
