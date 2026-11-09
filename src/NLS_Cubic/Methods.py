import scipy as sp
from scipy.sparse import diags, identity, bmat, kron, csc_matrix
from scipy.optimize import root
from scipy.fft import fft, ifft, dct, idct, dst, idst
from time import time

import sys
sys.path.append('../../')
from Base import *
from Utilities import *

i = 1j


class Focusing_NLS(NLS_numerical):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):

        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)
        self.l = self.dt/(self.h**2)
        self.order = order

        if self.border == "absorbing layer":
            cond_l_1 = "l_1" in self.pars.keys()
            cond_l_2 = "l_2" in self.pars.keys()
            cond_l_abs = "l_abs" in self.pars.keys()
            if not cond_l_1 or not cond_l_2 or not cond_l_abs:
                raise ValueError("Some absorbing parameters are missing")
            l_abs = self.pars["l_abs"]
            if l_abs < 0:
                raise ValueError("Absorbing layer can not be negative")

            self.right_layer = np.arange(self.L, l_abs, self.h)[1:]
            self.left_layer = -self.right_layer
            self.left_layer.sort()

            self.left_border_ind = len(self.left_layer)
            self.right_border_ind = self.left_border_ind + self.N - 1

            self.x_extended = np.append(self.x, self.right_layer)
            self.x_extended = np.append(self.left_layer, self.x_extended)

            self.N_extended = len(self.x_extended)
            
            self.x_original = self.x
            self.x = self.x_extended

            self.N_original = self.N
            self.N = self.N_extended

            if self.save_as_files is None:
                self.w = np.zeros((self.Nt, self.N_extended), dtype = 'complex')

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
        p = self.pars["p"]

        if self.save_as_files is not None and self.save_properties:
            if self.local:
                wn_r = self.w[np.abs(self.x-self.x_0)<self.L_tilde]

                e_r = self.h * np.sum(1/2 * norm(np.gradient(wn_r, self.h)) - 
                                p/4 * norm(wn_r)**2)

                wn_l = self.w[np.abs(self.x+self.x_0)<self.L_tilde]

                e_l = self.h * np.sum(1/2 * norm(np.gradient(wn_l, self.h)) - 
                                p/4 * norm(wn_l)**2)

                self.to_save_energy_R[self.save_iter] = e_r
                self.to_save_energy_L[self.save_iter] = e_l

            wn = self.w

            e = self.h * np.sum(1/2 * norm(np.gradient(wn, self.h)) - 
                                    p/4 * norm(wn)**2)

            self.to_save_energy[self.save_iter] = e

        else:
            energy = np.zeros(self.Nt)
            for n in range(self.Nt):
                wn = self.w[n]
                e = self.h * np.sum(1/2 * norm(np.gradient(wn, self.h)) - 
                                p/4 * norm(wn)**2)
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


class Time_CN(Focusing_NLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):

        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, order, local, print_iteration)

        self.l = self.dt/(self.h**2)
        if self.border == "absorbing layer":
            self.A = self.Ah_absorbing()
            self.I = identity(self.N_extended, format="csc")

        else:
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

        if self.border.lower() == "dirichlet":
            return A

        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = 1
            A[-1, 0] = 1

            A = A.tocsc()
            return A

        return A


    def Ah_absorbing(self):
        l_1 = self.pars["l_1"]
        l_2 = self.pars["l_2"]

        e = np.ones(self.N, dtype = 'complex') * -2

        e[self.left_border_ind] = i * l_1 * self.h**2/2 - 2
        e[self.right_border_ind] = i * l_2 * self.h**2/2 - 2

        e[:self.left_border_ind] = i * l_1 * self.h**2 - 2
        e[self.right_border_ind+1:] = i * l_2 * self.h**2 - 2

        f_1 = np.ones(self.N-1, dtype = 'complex')
        f_2 = np.ones(self.N-1, dtype = 'complex')

        #f_1[-1] = 2
        #f_2[0] = 2

        f_1[self.left_border_ind] = 1 + i * l_1 * self.h**2/12
        f_1[self.right_border_ind] = 1 - i * l_2 * self.h**2/12

        f_2[self.left_border_ind] = 1 - i * l_1 * self.h**2/12
        f_2[self.right_border_ind] = 1 + i * l_2 * self.h**2/12
        # Arreglo con las diagonales

        k_ = np.array([f_1,e,f_2], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales
    
        A = diags(k_, offset, format="csc", dtype='complex')

        return A


class CNFD(Time_CN):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, order, local, print_iteration)


    def F(self, u, v):
        p = self.pars["p"]
        w = u[:self.N]
        w_c = u[self.N:]
        i = 1j
        
        norm_w = norm(w)
        norm_v = norm(v)
        
        gp = self.dt * p/2 * (w + v) * (norm_w + norm_v)/2
        F_ = i * (v - w) + self.l/2 * self.A * (w + v) + gp
        F_conj = F_.conjugate()

        output = np.append(F_, F_conj)
        return output


    def F_prime(self, u, v):
        p = self.pars["p"]
        w = u[:self.N]
        w_c = u[self.N:]

        i = 1j

        norm_w = norm(w)
        norm_n = norm(v)

         # Arreglo con las diagonales
        k_dfdw = np.array([self.dt*p/4 * (2 * norm_w + v * w_c + norm_n)]) 
        k_dfdw_c = np.array([self.dt*p/4 *(w**2 + v * w)]) 
        offset = [0]

        dFdw_nl = diags(k_dfdw, offset, format="csc", dtype='complex')
        dFdw_c_nl = diags(k_dfdw, offset, format="csc", dtype='complex')
        
        dFdw = -i * self.I + self.l/2 * self.A + dFdw_nl
        dFdw_c = dFdw_c_nl
        dF_cdw = dFdw_c.conjugate()
        dF_cdw_c = dFdw.conjugate()

        jac = bmat([[dFdw, dFdw_c], [dF_cdw, dF_cdw_c]], format="csc", dtype="complex")
        return jac

    def G(self, u, v):
        i=1j

        p = self.pars["p"]
        
        norm_u = norm(u)
        norm_v = norm(v)

        linear = i * v + self.l/2 * self.A * v
        gp = self.dt*p/4 * (u + v) * (norm_u + norm_v)

        non_linear = linear + gp
        
        M_to_invert = i * self.I - self.l/2 * self.A
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


class CNFD_real(Time_CN):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.B = self.l/2 * self.B_A()
    
        # Array to save the solution
        self.w = np.zeros((self.Nt+1, 2*self.N))
    
    
        def set_w_theo(self, w_theo, eps=0, sigma=0):
            self.w_theo = w_theo
            
            #  Set init condition
            init = self.w_theo(self.x, self.ti) #self.w_theo.w(self.x, self.ti)
    
            if eps != 0:
                init[1:-1] += np.random.uniform(-1, 1, self.N-2)*eps
    
            if sigma != 0:
                init += np.exp(-sigma*self.x**2)
    
            self.w[0] = self.convert_to_array(init)
    
        def set_init_condition(self, w_0, eps=0, sigma=0):
            #  Set init condition
            init = w_0(self.x)
    
            if eps != 0:
                init[1:-1] += np.random.uniform(-1, 1, self.N-2)*eps
    
            if sigma != 0:
                init += np.exp(-sigma*self.x**2)
    
            self.w[0] = self.convert_to_array(init)
    
    
        def convert_to_array(self, w_complex):
            output = w_complex.view("(2,)float").T.flatten()
            return output
    
    
        def convert_to_complex(self):
            i = 1j
            x, y = self.w[:, :self.N], self.w[:, self.N:]
            self.w = x + i*y
    
        def F(self, u, v):
            xu, yu = u[:self.N], u[self.N:]
            xv, yv = v[:self.N], v[self.N:]
    
            norm_u = xu*xu + yu*yu
            norm_v = xv*xv + yv*yv
    
            # Arreglo con las diagonales
            k_ = np.array([norm_u+norm_v], dtype=object) 
            offset = [0] #Posiciones de las diagonales
    
            L = diags(k_, offset, format="csc", dtype='float64')
            anti_I_2 = np.array([[0, -1], [1, 0]])
            
            C_ = kron(anti_I_2, L)
            output = C_ * u
            return output
    
    
        def G(self, u, v):
            p = self.pars["p"]
            linear = self.B * u
    
            C = self.F(u, v)
            non_linear = self.dt*p/4 * C
            output = linear + non_linear
    
            return output
        
        def F1dx(self, x, y, x_0, y_0):
            p = self.pars["p"]
            
            e = 1 + self.dt * p/2 * x * (y + y_0)
            
            k_ = np.array([e], dtype=object) 
            offset = [0]
    
            output = diags(k_, offset, format="csc", dtype="float64")
            return output
        
        def F1dy(self, x, y, x_0, y_0):
            p = self.pars["p"]
            A = self.Ah()
            
            e_ = x * x + 3 * y * y + x_0 * x_0 + y_0 * y_0 + 2 * y * y_0
            e = (self.dt/4) * p * e_
            
            k_ = np.array([e], dtype=object) 
            offset = [0]
    
            D = diags(k_, offset, format="csc", dtype="float64")
            output = self.l/2 * self.A + D
            return output
        
        def F2dx(self, x, y, x_0, y_0):
            p = self.pars["p"]
            A = self.Ah()
            
            e_ = 3 * x * x + y * y + x_0 * x_0 + y_0 * y_0 + 2 * x * x_0
            e = (self.dt/4) * p * e_
            
            k_ = np.array([e], dtype=object) 
            offset = [0]
    
            D = diags(k_, offset, format="csc", dtype="float64")
            output = -self.l/2 * self.A -  D
            return output
        
        def F2dy(self, x, y, x_0, y_0):
            p = self.pars["p"]
            
            e = 1 - self.dt * p/2 * (x + x_0) * y
            
            k_ = np.array([e], dtype=object) 
            offset = [0]
    
            output = diags(k_, offset, format="csc", dtype="float64")
            return output
    
        
        def F_prime(self, X, X_0):
            x, y = X[:self.N], X[self.N:]
            x_0, y_0 = X_0[:self.N], X_0[self.N:]
            F1dx_ = self.F1dx(x, y, x_0, y_0)
            F1dy_ = self.F1dy(x, y, x_0, y_0)
            F2dx_ = self.F2dx(x, y, x_0, y_0)
            F2dy_ = self.F2dy(x, y, x_0, y_0)
            jac = bmat([[F1dx_, F1dy_], [F2dx_, F2dy_]], format="csc", dtype="float64")
            return jac
    
        def B_A(self):
            anti_I_2 = np.array([[0, -1], [1, 0]])
            B_ = kron(anti_I_2, self.A)
            return B_
    
    
        def solve(self, **kwargs):
            init_time = time()
            for n in range(self.Nt):
                if self.save_as_files is None:
                    self.w[n+1] = self.update(n, **kwargs)
    
                else:
                    self.w = self.update(n, **kwargs)
                    if Nelem is None:
                        save_file(self.w, self.save_as_files + f"{self.save_iter:010d}")
    
                    elif self.save_iter%Nelem==0:
                        save_file(self.w, self.save_as_files + f"{self.save_iter:010d}")
                
            final_time = time()
            self.time = final_time - init_time
            self.convert_to_complex()
    
    
        def update(self, n, **kwargs):
            i=1j
    
            if self.save_as_files is None:
                # Actual state
                wn = self.w[n]
    
            else:
                wn = self.w
            
            def F_(X):
                output = X - self.G(X, wn) - wn - self.G(wn, X)
                return output
    
            def F_prime_(X):
                output = self.F_prime(X, wn)
                return output
            
            # Update
            w_n_1, error, iteration = newton(wn, F_, F_prime_, self.h)
            if self.print_iteration:
                print("Iteration ",n+1, " of ", self.Nt, ". Error: ", error, ". Newton iterations: ", iteration)
    
            if error != error:
                self.stop = True
                pass
    
            return w_n_1


class SSFD(Time_CN):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)


    def C(self, x, y):
        p = self.pars["p"]
        
        e = self.dt * p/2 * (x**2 + y**2)
        
         # Arreglo con las diagonales
        k_ = np.array([e], dtype=object) 
        offset = [0] #Posiciones de las diagonales

        L = diags(k_, offset, format="csc", dtype='float64')
        anti_I_2 = np.array([[0, -1], [1, 0]])
        C_ = kron(anti_I_2, L)
        return C_


    def F(self, X):
        x, y = X[:self.N], X[self.N:]

        p = self.pars["p"]
        linear = self.B * X
    
        C = self.C(x, y)
        non_linear =  C * X
        output = linear + non_linear

        return output


    def F1dx(self, x, y):
        p = self.pars["p"]
        
        e = 1 + self.dt * p * x * y
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        output = diags(k_, offset, format="csc", dtype='float64')
        return output


    def F1dy(self, x, y):
        p = self.pars["p"]
        
        e = (self.dt/2) * p * (x**2 + 3 * y**2)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype='float64')
        output = self.l/2 * self.A + D
        return output


    def F2dx(self, x, y):
        p = self.pars["p"]
        
        e = (self.dt/2) * p * (3 * x**2 + y**2)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype='float64')
        output = -self.l/2 * self.A - D
        return output


    def F2dy(self, x, y):
        p = self.pars["p"]
        
        e = 1 - self.dt * p * x * y
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        output = diags(k_, offset, format="csc", dtype='float64')
        return output


    def F_prime(self, X):
        x, y = X[:self.N], X[self.N:]
        F1dx_ = self.F1dx(x, y)
        F1dy_ = self.F1dy(x, y)
        F2dx_ = self.F2dx(x, y)
        F2dy_ = self.F2dy(x, y)
        jac = bmat([[F1dx_, F1dy_], [F2dx_, F2dy_]], format="csc", dtype="float64")
        return jac


    def update(self, n, **kwargs):

        if self.save_as_files is None:
            # Actual state
            wn = self.w[n]

        else:
            wn = self.w
        
        def F_(X):
            output = X - wn - self.F(X)
            return output
        
        # Iterative method
        Z, error = newton(wn, F_, self.F_prime, tol=1e-2)
        wn_sig = 2 * Z - wn
        
        # Actualiza paso siguiente
        return wn_sig


class SemiImplicit(Focusing_NLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        if self.save_as_files is not None:
            raise ValueError("This method cannot save files")


    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo

        #  Set init condition
        self.w[0] = self.w_theo(self.x, self.ti)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        # Set for n=1
        p = self.pars["p"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * p * norm(self.w[0]))
        f_0 = np.ones(self.N-1, dtype = 'complex_') * i * self.l
        k_0 = np.array([f_0,e_0,f_0], dtype=object) 
        offset = [-1,0,1] # Diagonals positions

        B_0 = diags(k_0, offset, dtype='complex_')
        self.w[1] = B_0 * self.w[0]


    def set_init_condition(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0(self.x)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        # Set for n=1
        p = self.pars["p"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * p * norm(self.w[0]))
        f_0 = np.ones(self.N-1, dtype = 'complex_') * i * self.l
        k_0 = np.array([f_0,e_0,f_0], dtype=object) 
        offset = [-1,0,1] # Diagonals positions

        B_0 = diags(k_0, offset, dtype='complex_')
        self.w[1] = B_0 * self.w[0]
        
    
    def set_init_array(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        # Set for n=1
        p = self.pars["p"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * p * norm(self.w[0]))
        f_0 = np.ones(self.N-1, dtype = 'complex_') * i * self.l
        k_0 = np.array([f_0,e_0,f_0], dtype=object) 
        offset = [-1,0,1] # Diagonals positions

        B_0 = diags(k_0, offset, dtype='complex_')
        self.w[1] = B_0 * self.w[0]


    def Ah(self):
        p = self.pars["p"]
        e = np.ones(self.N, dtype = 'complex_') * (i - 2 * self.l)

        if self.border.lower() == "neumann":
            e[0] += self.l
            e[-1] += self.l 

        f = np.ones(self.N-1, dtype = 'complex_') * self.l


        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1]

        A = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return A


        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = self.l
            A[-1, 0] = self.l

            A = A.tocsc()
            return A

        return A
    
    def Bh(self, n):
        p = self.pars["p"]
        wn = self.w[n]
        B = identity(self.N, format="csc", dtype='complex_') * (-2 * p * self.dt * norm(wn))
        return B
    
    
    def update(self, n, **kwargs):
        # Actual and last step
        wn_last = self.w[n-1]
        wn = self.w[n]
        
        # A matrix
        A = self.Ah()

        # B Matriz
        B = self.Bh(n)
        b = B * wn
        
        # C Matrix
        I = identity(self.N, format="csc", dtype='complex_')
        C = 2 * i * I - A
        c = C * wn_last
        
        # Invierte matriz
        wn_sig = spsolve(A, b+c)
        
        # Actualiza paso siguiente
        self.w[n+1] = wn_sig


    def solve(self):
        init_time = time()
        for n in range(1, self.Nt):
            self.update(n)
        final_time = time()
        self.time = final_time - init_time


class Relaxation(Focusing_NLS):

    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.u = norm(self.w[0])
        self.method = "Relaxed"
        self.l = self.dt/(self.h**2)


    def set_w_theo(self, w_theo, eps=0, sigma=0):

        self.w_theo = w_theo

        #  Set init condition
        self.w[0] = self.w_theo(self.x, self.ti)
        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)
        self.u = norm(self.w[0])


    def set_init_condition(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0(self.x)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        self.u = norm(self.w[0])


    def set_init_array(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        self.u = norm(self.w[0])


    def Ah(self, n):
        p = self.pars["p"]
        e = np.ones(self.N, dtype = 'complex_') * (i - self.l + p * self.dt * self.u/2)

        if self.border.lower() == "neumann":
            e[0] += self.l/2
            e[-1] += self.l/2


        f = np.ones(self.N-1, dtype = 'complex_') * self.l/2

        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] 

        A = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return A


        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = self.l/2 
            A[-1, 0] = self.l/2

            A = A.tocsc()
            return A

        return A


    def Bh(self, n):
        p = self.pars["p"]
        e = np.ones(self.N, dtype = 'complex_') * (i + self.l - p * self.dt * self.u/2)

        if self.border.lower() == "neumann":
            e[0] += -self.l/2
            e[-1] += -self.l/2

        f = np.ones(self.N-1, dtype = 'complex_') * -self.l/2

        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1]

        B = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return B

        if self.border.lower() == "periodic":
            B = B.tolil()

            B[0, -1] = -self.l/2 
            B[-1, 0] = -self.l/2

            B.tocsc()
            return B

        return B
    
    
    def update(self, n, **kwargs):

        if self.save_as_files is None:
            # Actual state
            wn = self.w[n]

        else:
            wn = self.w

        self.u = 2 * norm(wn) - self.u
        
        # Obtiene matriz A
        A = self.Ah(n)

        # Matriz B
        B = self.Bh(n)
        b = B*wn
        
        # Solve linear problem
        wn_sig = spsolve(A, b)
        
        # Update
        return wn_sig


    def get_energy_relaxed(self, plot=False, path=None):
        p = self.pars["p"]
        energy = np.zeros(self.Nt)
        u = norm(self.w[0])
        for n in range(self.Nt):
            wn = self.w[n]
            u_sig = 2 * norm(wn) - u

            e_1 = np.sum(np.abs(np.diff(wn))**2/(self.h))
            e_2 = -self.h*p/4 * np.sum(u * u_sig)

            e = e_1 + e_2
            energy[n] = e
            u = u_sig
        
        if plot:
            plt.plot(self.t, energy, ".")#-energy[0]

            plt.xlabel(r"$t$")
            plt.ylabel(r"$E(t) $")#- E(0)
            plt.tight_layout()
            
            if path is not None:
                plt.savefig(path, dpi=300, bbox_inches = "tight")
            plt.show()

        return energy


class TSNLS(Focusing_NLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.l = self.dt/(self.h**2)


    @abstractmethod
    def method_update(self, w):
        pass


    def ODE_solution(self, w):
        p = self.pars["p"]
        i=1j
        output = np.exp((i*self.dt*p*(norm(w)))/2) * w
        return output


    def update(self, n, **kwargs):

        if self.save_as_files is None:
            # Actual state
            wn = self.w[n]

        else:
            wn = self.w

        w1 = self.ODE_solution(wn)

        w2 = self.method_update(w1)

        wn_sig = self.ODE_solution(w2)

        # Update
        return wn_sig


class TSFD(TSNLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.l = self.dt/(self.h**2)
    
    def Ah(self):
        i = 1j
        p = self.pars["p"]
        e = np.ones(self.N, dtype = 'complex_') * (1+self.l*i)

        if self.border.lower() == "neumann":
            e[0] -= i*self.l/2
            e[-1] -= i*self.l/2

        f = np.ones(self.N-1, dtype = 'complex_') * -i*self.l/2

        # Arreglo con las diagonales
        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales

        A = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return A


        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = -i*self.l/2
            A[-1, 0] = -i*self.l/2

            A = A.tocsc()
            return A

        return A


    def method_update(self, w):
        
        I = identity(self.N, format="csc", dtype='complex_')
        A = self.Ah()
        B = 2*I - A
        b = B*w
        
        w_ = spsolve(A, b)
        return w_


class TSSP(TSNLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

    def method_update(self, w):
        i = 1j
        mu_l = np.fft.fftfreq(self.N) * self.N * np.pi / (2*self.L)
        
        # Dirichlet has some problems
        if self.border.lower() == "dirichlet":
            w_ = idst(np.exp((-i*self.dt*(mu_l**2))) * dst(w))


        elif self.border.lower() == "neumann":
            w_ = idct(np.exp(-(i*self.dt*(mu_l**2))) * dct(w))

            
        elif self.border.lower() == "periodic":
            mu_l *= 2
            w_ = ifft(np.exp(-(i*self.dt*(mu_l**2))) * fft(w))

        return w_


class TSNLS_high(Focusing_NLS):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

        self.l = self.dt/(self.h**2)

        self.a1 = 1/(2*(2-2**(1/3)))#1/6 * (2 + 2**(1/3) + 2**(-1/3))
        self.a2 = (1-2**(1/3))/(2*(2-2**(1/3)))#1/6 * (1 - 2**(1/3) - 2**(-1/3))
        self.a3 = self.a2
        self.a4 = self.a1
        self.b1 = 1/(2-2**(1/3)) #b2
        self.b2 = -(2**(1/3))/(2-2**(1/3))#1/(1-2**(2/3)) #b3
        self.b3 = self.b1 #b4


    @abstractmethod
    def method_update(self, w, b):
        pass


    def set_boundary(self, n):
        pass


    def ODE_solution(self, w, a):
        p = self.pars["p"]
        i=1j
        output = np.exp((i*a*self.dt*p*(norm(w)))) * w
        return output


    def update(self, n):

        if self.save_as_files is None:
            # Actual state
            wn = self.w[n]

        else:
            wn = self.w
        
        w1 = self.ODE_solution(wn, self.a1) #a1
        
        w2 = self.method_update(w1, self.b1) #b2
        
        w3 = self.ODE_solution(w2, self.a2) #a2
        
        w4 = self.method_update(w3, self.b2) #b3
        
        w5 = self.ODE_solution(w4, self.a3) #a3
        
        w6 = self.method_update(w5, self.b3) #b4
        
        wn_sig = self.ODE_solution(w6, self.a4) #a4
        
        # Update
        return wn_sig


class TSFDH(TSNLS_high):
    
    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

    def Ah(self, b):
        i = 1j
        p = self.pars["p"]
        e = np.ones(self.N, dtype = 'complex_') * (1+b*self.l*i)

        if self.border.lower() == "neumann":
            e[0] -= i*b*self.l/2
            e[-1] -= i*b*self.l/2

        f = np.ones(self.N-1, dtype = 'complex_') * -i*b*self.l/2

        # Arreglo con las diagonales
        k_ = np.array([f,e,f], dtype=object) 
        offset = [-1,0,1] #Posiciones de las diagonales

        A = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return A


        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = -i*b*self.l/2
            A[-1, 0] = -i*b*self.l/2

            A = A.tocsc()
            return A

        return A

    
    def method_update(self, w, b):

        I = identity(self.N, format="csc", dtype='complex_')
        A = self.Ah(b)
        B = 2*I - A
        b = B*w
        
        w_ = spsolve(A, b)
        return w_


class TSSPH(TSNLS_high):

    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, order=1, local=False, print_iteration=True):
        super().__init__(L, N, Nt, tf, pars, border, x_0, L_tilde, ti, path, save_as_files, save_properties, local, print_iteration)

    def method_update(self, w, b):
        i = 1j
        mu_l = np.fft.fftfreq(self.N) * self.N * np.pi / (2*self.L)
        
        # Dirichlet has some problems
        if self.border.lower() == "dirichlet":
            w_ = idst(np.exp((-i*b*self.dt*(mu_l**2))) * dst(w))
            
        elif self.border.lower() == "neumann":
            w_ = idct(np.exp(-(i*b*self.dt*(mu_l**2))) * dct(w))
            
        elif self.border.lower() == "periodic":
            mu_l *= 2
            w_ = ifft(np.exp(-(i*b*self.dt*(mu_l**2))) * fft(w))

        return w_
