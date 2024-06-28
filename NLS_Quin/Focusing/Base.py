import numpy as np
from abc import ABC, abstractmethod


class NLS_FQ_solutions(ABC):

    def __init__(self, L, N, dt, tf, pars, ti=0):
        self.N = N
        self.L = L
        self.pars = pars
        
        self.dt = dt
        self.ti = ti
        self.tf = tf
        self.Nt = int((self.tf-self.ti)//self.dt)

        # Grids
        self.x = np.linspace(-self.L, self.L, self.N)
        self.t = np.linspace(self.ti, self.tf, self.Nt)

        
    def norm(self, w):
        return np.abs(w)**2


    @abstractmethod
    def animate_solution(path, N_frames):
        pass
    
    @abstractmethod
    def plot(self, t):
        pass

    @abstractmethod
    def plot_real(self, t):
        pass

    @abstractmethod
    def plot_imag(self, t):
        pass