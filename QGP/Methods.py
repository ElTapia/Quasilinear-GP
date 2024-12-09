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
        self.A_ = self.A_I()
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


    def A_I(self):
        I = np.array([[1, 0], [0, 1]])
        A_I_ = kron(I, self.A)
        return A_I_


class Real_CNFD(Time_CN):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.method = "CNFD"

        # Array to save the solution
        self.w = np.zeros((self.Nt+1, 2*self.N))


    def convert_to_array(self, w_complex):
        output = w_complex.view("(2,)float").T.flatten()
        return output


    def convert_to_complex(self):
        i = 1j
        x, y = self.w[:, :self.N], self.w[:, self.N:]
        self.w = x + i*y


    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo
        
        #  Set init condition
        init = self.w_theo(self.x, self.ti)

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
        
    
    def set_init_array(self, w_0, eps=0, sigma=0):
        #  Set init condition
        init = w_0

        if eps != 0:
            init[1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            init += np.exp(-sigma*self.x**2)

        self.w[0] = self.convert_to_array(init)


    def F(self, u, v):
        xu, yu = u[:self.N], u[self.N:]
        xv, yv = v[:self.N], v[self.N:]

        norm_u = xu*xu + yu*yu
        norm_v = xv*xv + yv*yv
        
        #F_1 = yu - yv + self.dt/2 * (xu + xv) - self.dt/4 * (xu + xv) * (norm_u + norm_v)
        #F_2 = xv - xu + self.dt/2 * (yu + yv) - self.dt/4 * (yu + yv) * (norm_u + norm_v)
        
        #Dyu = self.A * yu
        #Dyv = self.A * yv
        
        #Dxu = self.A * xu
        #Dxv = self.A * xv
        
        #F_1 = F_1 + self.l/2 * Dxu + self.l/2 * Dxv
        #F_2 = F_2 + self.l/2 * Dyu + self.l/2 * Dyv

        #output = np.append(F_1, F_2)
        
        #return output

    
        # Arreglo con las diagonales
        k_ = np.array([1 - (norm_u + norm_v)/2], dtype=object)
        offset = [0]

        L = diags(k_, offset, format="csc", dtype='float64')
    
        anti_I = np.array([[0, -1], [1, 0]])
        I = identity(2)
        
        C = kron(I, L) # anti_I

        return C


    def G(self, u, v):
        k = self.pars["k"]
        #sum_uv = u + v
        #linear = self.l/2 * self.B * sum_uv

        #L_1, L_2 = self.F(u, v)
        
        #cubic = self.dt/2 * L_2 * sum_uv

        #quasi = k * self.l/4 * (self.A_ * L_1) * u

        #output = linear + cubic #- quasi
        
        linear = self.l/2 * self.A_ * (u + v)

        C = self.F(u, v)
        non_linear = self.dt/2 * C * (u + v)

        UV = np.append((u-v)[self.N:], (v-u)[:self.N])
        
        if k==0:
            return UV + linear + non_linear
        
        xu, yu = u[:self.N], u[self.N:]
        xv, yv = v[:self.N], v[self.N:]

        norm_u = xu*xu + yu*yu
        norm_v = xv*xv + yv*yv
        
        A_norm = np.array([self.A * (norm_u + norm_v)])
        offset = [0]

        A_norm_diag = diags(A_norm, offset, format="csc", dtype='float64')
    
        anti_I = np.array([[0, -1], [1, 0]])
        
        A_norm_uv = kron(anti_I, A_norm_diag)
        
        quasi = k * self.l/4 * A_norm_uv * (u + v)

        output = linear + non_linear + quasi

        return output


    def F1dy(self, x, y, x_0, y_0):

        k = self.pars["k"]
        e = 1 - self.dt/2 * x * (y + y_0)

        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")
        
        if k == 0:
            return D

        Q = k * self.l/2 * ((self.A.toarray() * x[np.newaxis, :]) * (y + y_0)[:, np.newaxis])

        output = D + Q

        return sp.sparse.csc_matrix(output)


    def F1dx(self, x, y, x_0, y_0):

        k = self.pars["k"]
        I = identity(len(x))

        e_ = y * y + 3 * x * x + x_0 * x_0 + y_0 * y_0 + 2 * x * x_0

        k_ = np.array([e_], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")

        if k == 0:
            output = self.l/2 * self.A + self.dt/2 * I - self.dt/4 * D
            return sp.sparse.csc_matrix(output)
    
        f_ = 2 * y[1:] * y[:-1]
        e_ = -2 * (x * x + 3 * y * y + x_0 * x_0 + y_0 * y_0)
        
        # Neumann boundary
        e_[0] = e_[0] + x[0] * x[0] + 3 * y[0] * y[0] + x_0[0] * x_0[0] + y_0[0] * y_0[0]
        e_[-1] = e_[-1] + x[-1] * x[-1] + 3 * y[-1] * y[-1]  + x_0[-1] * x_0[-1] + y_0[-1] * y_0[-1]
        
        # Other derivative terms
        mod_xy = x * x + y * y
        mod_xy_0 = x_0 * x_0 + y_0 * y_0

        roll_xy = np.roll(mod_xy, 1)
        roll_xy[0] = 0
        
        roll_xy_0 = np.roll(mod_xy_0, 1)
        roll_xy_0[0] = 0

        aroll_xy = np.roll(mod_xy, -1)
        aroll_xy[-1] = 0
        
        aroll_xy_0 = np.roll(mod_xy_0, -1)
        aroll_xy_0[-1] = 0

        sum_mod_xy = roll_xy + aroll_xy
        sum_mod_xy_0 = roll_xy_0 + aroll_xy_0

        e_ = e_ + sum_mod_xy + sum_mod_xy_0
    
        k_ = np.array([f_, e_, f_], dtype=object) 
        offset = [-1, 0, 1]

        E_ = diags(k_, offset, format="csc", dtype="float64")
        
        G = 2 * ((self.A.toarray() * y[np.newaxis, :]) * y_0[:, np.newaxis])
        
        output = -self.l/2 * self.A - self.dt/2 * I + self.dt/4 * D + k * self.l/4 * (E_ + G)
        return sp.sparse.csc_matrix(output)


    def F2dy(self, x, y, x_0, y_0):
        k = self.pars["k"]
        
        I = identity(len(x))
        
        e_ = 3 * y * y + x * x + x_0 * x_0 + y_0 * y_0 + 2 * y * y_0

        k_ = np.array([e_], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")
        
        if k==0:
            output = self.l/2 * self.A + self.dt/2 * I - self.dt/4 * D
            return sp.sparse.csc_matrix(output)

        f_ = 2 * x[1:] * x[:-1]
        e_ = -2 * (3 * x * x + y * y + x_0 * x_0 + y_0 * y_0)
        
        # Neumann boundary
        e_[0] = e_[0] + 3 * x[0] * x[0] + y[0] * y[0] + x_0[0] * x_0[0] + y_0[0] * y_0[0]
        e_[-1] = e_[-1] + 3 * x[-1] * x[-1] + y[-1] * y[-1]  + x_0[-1] * x_0[-1] + y_0[-1] * y_0[-1]
        
        # Other derivative terms
        mod_xy = x * x + y * y
        mod_xy_0 = x_0 * x_0 + y_0 * y_0

        roll_xy = np.roll(mod_xy, 1)
        roll_xy[0] = 0
        
        roll_xy_0 = np.roll(mod_xy_0, 1)
        roll_xy_0[0] = 0

        aroll_xy = np.roll(mod_xy, -1)
        aroll_xy[-1] = 0
        
        aroll_xy_0 = np.roll(mod_xy_0, -1)
        aroll_xy_0[-1] = 0

        sum_mod_xy = roll_xy + aroll_xy
        sum_mod_xy_0 = roll_xy_0 + aroll_xy_0

        e_ = e_ + sum_mod_xy + sum_mod_xy_0
    
        k_ = np.array([f_, e_, f_], dtype=object) 
        offset = [-1, 0, 1]

        E_ = diags(k_, offset, format="csc", dtype="float64")
        
        G = 2 * ((self.A.toarray() * x[np.newaxis, :]) * x_0[:, np.newaxis])
        
        output = self.l/2 * self.A + self.dt/2 * I - self.dt/4 * D - k * self.l/4 * (E_ + G)
        return sp.sparse.csc_matrix(output)
    
    def F2dx(self, x, y, x_0, y_0):
        k = self.pars["k"]
        
        e = -1 + self.dt/2 * y * (x + x_0)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")
        
        if k == 0:
            return D

        Q = k * self.l/2 * ((self.A.toarray() * y[np.newaxis, :]) * (x + x_0)[:, np.newaxis])

        output = D - Q

        return sp.sparse.csc_matrix(output)


    def F_prime(self, X, X_0):
        x, y = X[:self.N], X[self.N:]
        x_0, y_0 = X_0[:self.N], X_0[self.N:]
        F1dx_ = self.F1dx(x, y, x_0, y_0)
        F1dy_ = self.F1dy(x, y, x_0, y_0)
        F2dx_ = self.F2dx(x, y, x_0, y_0)
        F2dy_ = self.F2dy(x, y, x_0, y_0)
        jac = bmat([[F1dx_, F1dy_], [F2dx_, F2dy_]], format="csc", dtype="float64")
        return jac


    def update(self, n):
        i=1j
        # Actual state
        wn = self.w[n]

        def F_(X):
            output = self.G(X, wn)
            return output

        def F_prime_(X):
            output = self.F_prime(X, self.w[n])
            return output

        # Update
        w_n_1, error = newton(wn, F_, F_prime_)
        self.w[n+1] = w_n_1
        #print("Newton error: ", error)


    def solve(self):
        init_time = time()
        for n in range(self.Nt):
            self.update(n)
        final_time = time()
        self.time = final_time - init_time
        self.convert_to_complex()


class Complex_CNFD(Time_CN):
    
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
        
        gp = self.dt/2 * (w + v) * (1 - (norm_w + norm_v)/2)
        quasi = k * self.dt/4 * (w + v) * (self.A * (norm_w + norm_v))
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
        
        dFdw = (-i + self.dt/2) * self.I + self.l/2 * self.A - dFdw_nl + k * self.dt/4 * quasi_dFdw
        dFdw_c = -dFdw_c_nl + k * self.dt/4 * quasi_dFdw_c
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