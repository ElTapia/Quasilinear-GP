import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod

from Base import NLS_solutions


class NLS_solitons(NLS_solutions):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = ""
        c = self.pars["c"]
        self.A = np.sqrt((2-c**2)/2)

        self.w = self.theo
    
    @abstractmethod
    def theo(self, x, t):
        pass
    
    def plot(self, t):
        c = self.pars["c"]
        plt.plot(self.x, self.norm(self.w(self.x, t)), label="c={}".format(round(c, 2)))

    def plot_real(self, t):
        c = self.pars["c"]
        plt.plot(self.x, np.real(self.w(self.x, t)), label="c={}".format(round(c, 2)))

    def plot_imag(self, t):
        c = self.pars["c"]
        plt.plot(self.x, np.imag(self.w(self.x, t)), label="c={}".format(round(c, 2)))

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


class NLS_dark_soliton(NLS_solitons):

    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = "Dark soliton"


    def theo(self, x, t):
        i = 1j
        c = self.pars["c"]
        x_0 = self.pars["x_0"]
        def v(x):
            sum_1 = self.A * np.tanh(np.sqrt(2-c**2)/2 * x)
            sum_2 = i*c/np.sqrt(2)
            return sum_1 + sum_2
        
        return v(x - x_0 - c*t)