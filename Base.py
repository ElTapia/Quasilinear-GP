import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from abc import ABC, abstractmethod
from Utilities import *

# Plot style parameters
plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['legend.fontsize'] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["axes.formatter.useoffset"]= False

class NLS_solutions(ABC):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        i = 1j
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


class NLS_numerical(NLS_solutions):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):

        super().__init__(L, N, dt, tf, pars, ti)
        # Exection time
        self.time = 0

        # Step sizes and total points
        self.h = 2*L/self.N

        # Parameters
        self.border = border

        # Array to save the solution and theoretical solution
        self.w = np.zeros((self.Nt+1, self.N), dtype = 'complex_')


    @abstractmethod
    def update(self):
        pass


    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo

        #  Set init condition with theo class
        self.w[0] = self.w_theo(self.x, self.ti)
        if eps != 0:
            self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)


    def set_init_condition(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0(self.x)
        if eps != 0:
            self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)
            
    
    def set_init_array(self, w_0, eps=0, sigma=0):
        #  Set with given init array
        self.w[0] = w_0
        if eps != 0:
            self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)


    def solve(self):
        init_time = time()
        for n in range(self.Nt):
            self.update(n)
        final_time = time()
        self.time = final_time - init_time


    def get_error(self, tf):
        n = int((tf - self.ti)//self.dt)
        w_sim = self.w[n]
        w_theo = self.w_theo(self.x, tf)
        
        diff = norm(w_sim - w_theo)
        sum_ = np.sum(diff)
        output = np.sqrt(self.h * sum_)
        return output


    # Define mass for all n
    def get_mass(self, plot=False, path=None):
        mass = np.zeros(self.Nt)
        for n in range(self.Nt):
            m = self.h * np.sum(norm(self.w[n]))
            mass[n] = md
            
        if plot:
            plt.plot(self.t, mass-mass[0], ".")

            plt.xlabel(r"$t$")
            plt.ylabel(r"$N(t)-N(0)$")
            plt.tight_layout()

            if path is not None:
                plt.savefig(path, dpi=300, bbox_inches = "tight")
            plt.show()

        return mass


    def get_execution_time(self):
        return self.time


    def plot(self, t, marker="", markersize=5, label=None):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, norm(self.w[n]), marker, markersize=markersize, label=label)


    def plot_real(self, t, label=None):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, np.real(self.w[n]), label=label)

    def plot_imag(self, t, label=None):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, np.imag(self.w[n]), label=label)


    def animate_solution(self, path, N_frames, theo=False, coord="norm"):
        # Create figure
        fig, ax = plt.subplots()

        # For empty plot
        empty_line = np.full(len(self.x), fill_value=None)

        # Plot init instant
        if coord.lower() == "norm":
            line1, = ax.plot(self.x, norm(self.w[0]), ls="--", label="Simulation")
        
        elif coord.lower() == "real":
            line1, = ax.plot(self.x, np.real(self.w[0]), ls="--", label="Simulation")
        
        elif coord.lower() == "imag":
            line1, = ax.plot(self.x, np.imag(self.w[0]), ls="--", label="Simulation")


        line2, = ax.plot(self.x, empty_line, label="Theo", alpha=.6)
        # Plotting with theo
        if theo and coord.lower() == "norm":
            line2.set_ydata(norm(self.w_theo(self.x, self.ti)))
                
        elif theo and coord.lower() == "real":
            line2.set_ydata(np.real(self.w_theo(self.x, self.ti)))
            
        elif theo and coord.lower() == "imag":
            line2.set_ydata(np.imag(self.w_theo(self.x, self.ti)))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|w|^2$")
        fig.tight_layout()

        # Animate function
        def animate(j, theo):

            # Update plot
            
            if coord.lower() == "norm":
                ax.set_ylim(min(norm(self.w[j]))-.2, max(norm(self.w[j]))+.2)
                #ax.set_ylim(-0.1, 17)
                line1.set_data(self.x, norm(self.w[j]))
                
            if coord.lower() == "real":
                ax.set_ylim(min(np.real(self.w[j]))-.2, max(np.real(self.w[j]))+.2)
                line1.set_data(self.x, np.real(self.w[j]))
                
            if coord.lower() == "imag":
                ax.set_ylim(min(np.imag(self.w[j]))-.2, max(np.imag(self.w[j]))+.2)
                line1.set_data(self.x, np.imag(self.w[j]))

            if theo and coord.lower() == "norm":
                line2.set_data(self.x, norm(self.w_theo(self.x, self.ti + j*self.dt)))
                ax.legend()
                
            if theo and coord.lower() == "real":
                line2.set_data(self.x, np.real(self.w_theo(self.x, self.ti + j*self.dt)))
                ax.legend()

            if theo and coord.lower() == "imag":
                line2.set_data(self.x, np.imag(self.w_theo(self.x, self.ti + j*self.dt)))
                ax.legend()

            return line1, line2,

        # Animación
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.Nt, N_frames), interval=150, fargs=[theo])

        ani.save(path)
        plt.show()
        return ani
    
    def get_colormap(self, path=None):
        fig, ax = plt.subplots(1, 1)
        im = ax.pcolorfast(self.x, self.t, norm(self.w))
        
        fig.colorbar(im)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        
        if path is not None:
            fig.savefig(path)
        plt.show()

