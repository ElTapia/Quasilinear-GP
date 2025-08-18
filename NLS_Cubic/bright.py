import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod

from Base import NLS_solutions

class NLS_soliton(NLS_solutions):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = ""
        
        self.w = self.theo

    @abstractmethod
    def theo(self, x, t):
        pass
    
    def plot(self, t):
        plt.plot(self.x, self.norm(self.w(self.x, t)))

    def plot_real(self, t):
        plt.plot(self.x, np.real(self.w(self.x, t)))

    def plot_imag(self, t):
        plt.plot(self.x, np.imag(self.w(self.x, t)))

    def animate_solution(self, path, N_frames):
        # Create figure
        fig, ax = plt.subplots()
        
        # For empty plot
        empty_line = np.full(len(self.x), fill_value=None)

        # Plot init instant
        line1, = ax.plot(self.x, self.norm(self.w(self.x, self.ti)))
    
        ax.set_title("NLS solution "+self.soliton)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|w|^2$")

        # Animate function
        def animate(j):

            # Update plot
            line1.set_data(self.x, self.norm(self.w(self.x, self.ti+j*self.dt)))
            return line1, 

        # Animación
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.Nt, N_frames), interval=150)

        ani.save(path)
        return ani


class NLS_1_soliton(NLS_soliton):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = "1-soliton"
        
    def theo(self, x, t):
        i = 1j

        # Get parameters
        k = self.pars["k"]
        A = self.pars["A"]
        B = self.pars["B"]
        C_1 = self.pars["C_1"]
        C_2 = self.pars["C_2"]

        # Do calculations
        cte = A * np.sqrt(2/k)
        num = np.exp(i*B*x+i*(A**2-B**2)*t+i*C_1)
        den = np.cosh(A*x-2*A*B*t+C_2)

        return cte * num / den


class NLS_2_soliton(NLS_soliton):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = "2-soliton"

    # 2-soliton theoretical solution
    def theo(self, x, t):
        # Get parameters
        i = 1j
        k = self.pars["k"]

        a_1 = self.pars["a_1"]
        b_1 = self.pars["b_1"]
        a_2 = self.pars["a_2"]
        b_2 = self.pars["b_2"]

        l_1 = a_1 + i*b_1
        l_2 = a_2 + i*b_2
        l_1_bar = np.conjugate(l_1)
        l_2_bar = np.conjugate(l_2)

        l = l_1 + l_2_bar
        l_bar = np.conjugate(l)
        
        eta_1_0 = self.pars["eta_1_0"]
        eta_2_0 = self.pars["eta_2_0"]

        
        
        def F(x, t):
            eta_1 = l_1 * x + (i*(l_1**2)) * t + eta_1_0
            eta_1_bar = np.conjugate(eta_1)
            eta_2 = l_2 * x + (i*(l_2**2)) * t + eta_2_0
            eta_2_bar = np.conjugate(eta_2)

            eta = eta_1 + np.conjugate(eta_2)
            eta_bar = np.conjugate(eta)
        
            # Do calculations
            sum_1 = (l_1 + l_1_bar)**(-2) * np.exp(eta_1 + np.conjugate(eta_1))
            sum_2 = l**(-2) * np.exp(eta)
            sum_3 = l_bar**(-2) * np.exp(eta_bar)
            sum_4 = (l_2 + l_2_bar)**(-2) * np.exp(eta_2 + np.conjugate(eta_2))
            
            mult_1 = ((l_1 - l_2)**2) * ((l_1_bar - l_2_bar)**2) * ((l_1 + l_1_bar) * (l_2 + l_2_bar) * l * l_bar)**(-2)
            mult_2 = np.exp(eta_1 + eta_1_bar + eta_2 + eta_2_bar)
            sum_5 =  mult_1 * mult_2
            
            output = 1 + sum_1 + sum_2 + sum_3 + sum_4 + sum_5
            return output


        def G(x, t):
            eta_1 = l_1 * x + (i*(l_1**2)) * t + eta_1_0
            eta_1_bar = np.conjugate(eta_1)
            eta_2 = l_2 * x + (i*(l_2**2)) * t + eta_2_0
            eta_2_bar = np.conjugate(eta_2)

            eta = eta_1 + np.conjugate(eta_2)
            eta_bar = np.conjugate(eta)
        
            # Do calculations
            sum_1 = np.exp(eta_1)
            sum_2 = np.exp(eta_2)
            mult = (l_1 - l_2)**2
            mult_sum_1 = (l_bar * (l_1 + l_1_bar))**(-2) * np.exp(eta_1 + eta_2 + eta_1_bar)
            mult_sum_2 = (l * (l_2 + l_2_bar))**(-2) * np.exp(eta_2 + eta_1 + eta_2_bar)
            output = sum_1 + sum_2 + mult * (mult_sum_1 + mult_sum_2)
            return output
        
        # Constant
        cte = np.sqrt(2/k)
        
        return cte * G(x, t)/F(x, t)#cte * (num / den) * (sum_1 + sum_2) 
