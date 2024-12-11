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

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)

        # Array to save the solution
        self.w = np.zeros((self.Nt+1, self.N), dtype="complex_")


    def get_energy(self, plot=False, path=None):
        k = self.pars["k"]
        energy = np.zeros(self.Nt)
        for n in range(self.Nt):
            wn = self.w[n]
            e = np.sum(norm(np.diff(wn))/(2*self.h) - (self.h * k * (norm(wn[:-1]))**2)/4)
            energy[n] = e

        if plot:
            plt.plot(self.t, energy-energy[0], ".")

            plt.title("Energy "+self.method+" "+title)
            plt.xlabel(r"$t$")
            plt.ylabel(r"$E(t)-E(0)$")#-E(0)
            plt.tight_layout()

            if path is not None:
                plt.savefig(path, dpi=300, bbox_inches = "tight")
            plt.show()

        return energy


class Time_CN(Focusing_NLS):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)

        self.A = self.Ah()
        
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
    
    
    def Ah(self):
        e = np.ones(self.N, dtype = 'float64') * -2

        if self.border.lower() == "neumann":
            e[0] += 1
            e[-1] += 1

        f = np.ones(self.N-1, dtype = 'float64')

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


    def B_A(self):
        anti_I_2 = np.array([[0, -1], [1, 0]])
        B_ = kron(anti_I_2, self.A)
        return B_

    def solve(self):
        init_time = time()
        for n in range(self.Nt):
            self.update(n)
        final_time = time()
        self.time = final_time - init_time
        self.convert_to_complex()


class CNFD(Time_CN):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)


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
        k = self.pars["k"]
        linear = self.B * u

        C = self.F(u, v)
        non_linear = self.dt*k/4 * C
        output = linear + non_linear

        return output
    
    def F1dx(self, x, y, x_0, y_0):
        k = self.pars["k"]
        
        e = 1 + self.dt * k/2 * x * (y + y_0)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        output = diags(k_, offset, format="csc", dtype="float64")
        return output
    
    def F1dy(self, x, y, x_0, y_0):
        k = self.pars["k"]
        A = self.Ah()
        
        e_ = x * x + 3 * y * y + x_0 * x_0 + y_0 * y_0 + 2 * y * y_0
        e = (self.dt/4) * k * e_
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")
        output = self.l/2 * self.A + D
        return output
    
    def F2dx(self, x, y, x_0, y_0):
        k = self.pars["k"]
        A = self.Ah()
        
        e_ = 3 * x * x + y * y + x_0 * x_0 + y_0 * y_0 + 2 * x * x_0
        e = (self.dt/4) * k * e_
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype="float64")
        output = -self.l/2 * self.A -  D
        return output
    
    def F2dy(self, x, y, x_0, y_0):
        k = self.pars["k"]
        
        e = 1 - self.dt * k/2 * (x + x_0) * y
        
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


    def update(self, n):
        i=1j
        # Actual state
        wn = self.w[n]
        
        def F_(X):
            output = X - self.G(X, wn) - wn - self.G(wn, X)
            return output

        def F_prime_(X):
            output = self.F_prime(X, wn)
            return output
        
        # Update
        self.w[n+1], error = newton(wn, F_, F_prime_)


class SSFD(Time_CN):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)


    def C(self, x, y):
        k = self.pars["k"]
        
        e = self.dt * k/2 * (x**2 + y**2)
        
         # Arreglo con las diagonales
        k_ = np.array([e], dtype=object) 
        offset = [0] #Posiciones de las diagonales

        L = diags(k_, offset, format="csc", dtype='float64')
        anti_I_2 = np.array([[0, -1], [1, 0]])
        C_ = kron(anti_I_2, L)
        return C_


    def F(self, X):
        x, y = X[:self.N], X[self.N:]

        k = self.pars["k"]
        linear = self.B * X
    
        C = self.C(x, y)
        non_linear =  C * X
        output = linear + non_linear

        return output


    def F1dx(self, x, y):
        k = self.pars["k"]
        
        e = 1 + self.dt * k * x * y
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        output = diags(k_, offset, format="csc", dtype='float64')
        return output


    def F1dy(self, x, y):
        k = self.pars["k"]
        
        e = (self.dt/2) * k * (x**2 + 3 * y**2)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype='float64')
        output = self.l/2 * self.A + D
        return output


    def F2dx(self, x, y):
        k = self.pars["k"]
        
        e = (self.dt/2) * k * (3 * x**2 + y**2)
        
        k_ = np.array([e], dtype=object) 
        offset = [0]

        D = diags(k_, offset, format="csc", dtype='float64')
        output = -self.l/2 * self.A - D
        return output


    def F2dy(self, x, y):
        k = self.pars["k"]
        
        e = 1 - self.dt * k * x * y
        
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


    def update(self, n):
        # Paso actual y anterior
        wn = self.w[n]
        
        def F_(X):
            output = X - wn - self.F(X)
            return output
        
        # Iterative method
        Z, error = newton(wn, F_, self.F_prime, tol=1e-2)
        wn_sig = 2 * Z - wn
        
        # Actualiza paso siguiente
        self.w[n+1] = wn_sig


class SemiImplicit(Focusing_NLS):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)


    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo

        #  Set init condition
        self.w[0] = self.w_theo(self.x, self.ti)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        # Set for n=1
        k = self.pars["k"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * k * norm(self.w[0]))
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
        k = self.pars["k"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * k * norm(self.w[0]))
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
        k = self.pars["k"]
        e_0 = np.ones(self.N, dtype = 'complex_') * (1 - 2 * i * self.l + i * self.dt * k * norm(self.w[0]))
        f_0 = np.ones(self.N-1, dtype = 'complex_') * i * self.l
        k_0 = np.array([f_0,e_0,f_0], dtype=object) 
        offset = [-1,0,1] # Diagonals positions

        B_0 = diags(k_0, offset, dtype='complex_')
        self.w[1] = B_0 * self.w[0]


    def Ah(self):
        k = self.pars["k"]
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
        k = self.pars["k"]
        wn = self.w[n]
        B = identity(self.N, format="csc", dtype='complex_') * (-2 * k * self.dt * norm(wn))
        return B
    
    
    def update(self, n):
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

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
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
        k = self.pars["k"]
        e = np.ones(self.N, dtype = 'complex_') * (i - self.l + k * self.dt * self.u/2)

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
        k = self.pars["k"]
        e = np.ones(self.N, dtype = 'complex_') * (i + self.l - k * self.dt * self.u/2)

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
    
    
    def update(self, n):
        wn = self.w[n]

        self.u = 2 * norm(wn) - self.u
        
        # Obtiene matriz A
        A = self.Ah(n)

        # Matriz B
        B = self.Bh(n)
        b = B*wn
        
        # Solve linear problem
        wn_sig = spsolve(A, b)
        
        # Update
        self.w[n+1] = wn_sig


    def get_energy_relaxed(self, plot=False, path=None):
        k = self.pars["k"]
        energy = np.zeros(self.Nt)
        u = norm(self.w[0])
        for n in range(self.Nt):
            wn = self.w[n]
            u_sig = 2 * norm(wn) - u

            e_1 = np.sum(np.abs(np.diff(wn))**2/(self.h))
            e_2 = -self.h*k/4 * np.sum(u * u_sig)

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
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)


    @abstractmethod
    def method_update(self, w):
        pass


    def ODE_solution(self, w):
        k = self.pars["k"]
        i=1j
        output = np.exp((i*self.dt*k*(norm(w)))/2) * w
        return output


    def update(self, n):
        
        # Set boundary conditions
        #self.set_boundary(n)

        wn = self.w[n]

        w1 = self.ODE_solution(wn)

        w2 = self.method_update(w1)

        wn_sig = self.ODE_solution(w2)

        # Update
        self.w[n+1] = wn_sig


class TSFD(TSNLS):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.l = self.dt/(self.h**2)
    
    def Ah(self):
        i = 1j
        k = self.pars["k"]
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
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)

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
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
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
        k = self.pars["k"]
        i=1j
        output = np.exp((i*a*self.dt*k*(norm(w)))) * w
        return output


    def update(self, n):

        # Set boundary conditions
        #self.set_boundary(n)

        wn = self.w[n]
        
        w1 = self.ODE_solution(wn, self.a1) #a1
        
        w2 = self.method_update(w1, self.b1) #b2
        
        w3 = self.ODE_solution(w2, self.a2) #a2
        
        w4 = self.method_update(w3, self.b2) #b3
        
        w5 = self.ODE_solution(w4, self.a3) #a3
        
        w6 = self.method_update(w5, self.b3) #b4
        
        wn_sig = self.ODE_solution(w6, self.a4) #a4
        
        # Update
        self.w[n+1] = wn_sig


class TSFDH(TSNLS_high):
    
    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
    
    def Ah(self, b):
        i = 1j
        k = self.pars["k"]
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

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)


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
