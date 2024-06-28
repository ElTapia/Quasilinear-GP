from Base import NLS_FQ_solutions
from time import time
from matplotlib import animation
from abc import abstractmethod
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.fft import fft, ifft, dct, idct, dst, idst
import matplotlib.pyplot as plt
import numpy as np


class NLS_FQ_numerical(NLS_FQ_solutions):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):

        super().__init__(L, N, dt, tf, pars, ti)

        # Execution time
        self.time = 0

        # Step sizes and total points
        self.h = 2*L/self.N

        # Parameters
        self.border = border
        self.method = ""

        # Array to save the solution and theoretical solution
        self.w = np.zeros((self.Nt, self.N), dtype='complex_')

    @abstractmethod
    def update(self):
        pass

    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo

        #  Set init condition
        self.w[0] = self.w_theo.w(self.x, self.ti)
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

    def solve(self):
        init_time = time()
        e = 0.1
        for n in range(self.Nt-1):

            # Gets to the border
            if self.norm(self.w[n, 0]) >= e or self.norm(self.w[n, -1]) >= e:
                tf = n * self.dt
                self.tf = tf
                print("Final time changed to {}".format(tf))
                self.w = self.w[:n]
                self.t = self.t[:n]
                self.Nt = n
                break

            self.update(n)
        final_time = time()
        self.time = final_time - init_time

    def get_error(self, tf):
        n = int((tf - self.ti)//self.dt)
        w_sim = self.w[n]
        w_theo = self.w_theo.w(self.x, tf)

        diff = np.abs(w_sim - w_theo)**2
        sum_ = np.sum(diff)
        output = np.sqrt(self.h * sum_)
        return output

    def get_virial(self, path, title, plot=True):
        virial = np.zeros(self.Nt)
        for n in range(self.Nt):
            wn = self.w[n]
            v_1 = self.norm(np.diff(wn))/(self.h)
            v_2 = (self.h * (self.norm(wn[1:]))**2)/4
            v_3 = (self.h * 0.04*(self.norm(wn[1:]))**3)/3
            v = np.sum(v_1 - v_2 - v_3)
            virial[n] = v

        if plot:
            plt.plot(self.t, virial)
            plt.hlines(self.t[0], self.t[-1], 0)

            plt.title("Virial "+title)
            plt.xlabel(r"$t$")
            plt.ylabel(r"$V(t)$")
            plt.tight_layout()

            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.show()
        return virial

    def get_execution_time(self):
        return self.time

    def plot(self, t, marker="", markersize=5):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, self.norm(self.w[n]), marker, markersize=markersize)

    def plot_real(self, t):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, np.real(self.w[n]))

    def plot_imag(self, t):
        n = int((t - self.ti)//self.dt)
        plt.plot(self.x, np.imag(self.w[n]))

    def animate_solution(self, path, N_frames, theo=False, coord="norm"):
        # Create figure
        fig, ax = plt.subplots()

        # For empty plot
        empty_line = np.full(len(self.x), fill_value=None)

        # Plot init instant
        if coord.lower() == "norm":
            line1, = ax.plot(self.x, self.norm(self.w[0]),
                             ls="--", label="Simulation")

        elif coord.lower() == "real":
            line1, = ax.plot(self.x, np.real(self.w[0]),
                             ls="--", label="Simulation")

        elif coord.lower() == "imag":
            line1, = ax.plot(self.x, np.imag(self.w[0]),
                             ls="--", label="Simulation")

        line2, = ax.plot(self.x, empty_line, label="Theo", alpha=.6)
        # Plotting with theo
        if theo and coord.lower() == "norm":
            line2.set_ydata(self.norm(self.w_theo.w(self.x, self.ti)))

        elif theo and coord.lower() == "real":
            line2.set_ydata(np.real(self.w_theo.w(self.x, self.ti)))

        elif theo and coord.lower() == "imag":
            line2.set_ydata(np.imag(self.w_theo.w(self.x, self.ti)))

        ax.set_title("NLS-FQ solution")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|w|^2$")
        fig.tight_layout()

        # Animate function
        def animate(j, theo):

            # Update plot

            if coord.lower() == "norm":
                ax.set_ylim(min(self.norm(self.w[j]))-.2,
                            max(self.norm(self.w[j]))+.2)
                # ax.set_ylim(-0.1, 17)
                line1.set_data(self.x, self.norm(self.w[j]))

            if coord.lower() == "real":
                ax.set_ylim(min(np.real(self.w[j]))-.2,
                            max(np.real(self.w[j]))+.2)
                line1.set_data(self.x, np.real(self.w[j]))

            if coord.lower() == "imag":
                ax.set_ylim(min(np.imag(self.w[j]))-.2,
                            max(np.imag(self.w[j]))+.2)
                line1.set_data(self.x, np.imag(self.w[j]))

            if theo and coord.lower() == "norm":
                w = self.w_theo.w
                line2.set_data(self.x,
                               self.norm(w(self.x, self.ti + j*self.dt)))
                ax.legend()

            if theo and coord.lower() == "real":
                w = self.w_theo.w
                line2.set_data(self.x,
                               np.real(w(self.x, self.ti + j*self.dt)))
                ax.legend()

            if theo and coord.lower() == "imag":
                w = self.w_theo.w
                line2.set_data(self.x,
                               np.imag(w(self.x, self.ti + j*self.dt)))
                ax.legend()

            return line1, line2,

        # Animación
        ani = animation.FuncAnimation(fig, animate,
                                      frames=np.arange(0, self.Nt, N_frames),
                                      interval=150, fargs=[theo])

        ani.save(path)
        plt.show()
        return ani

    def get_colormap(self, path):
        fig, ax = plt.subplots(1, 1)
        im = ax.pcolorfast(self.x, self.t, self.norm(self.w))

        fig.colorbar(im)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_title("NLS-FQ solution")

        fig.savefig(path)
        plt.show()


class Relax(NLS_FQ_numerical):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.u = self.norm(self.w[0])
        self.method = "Relaxed"
        self.la = self.dt/(self.h**2)

    def set_w_theo(self, w_theo, eps=0, sigma=0):

        self.w_theo = w_theo

        #  Set init condition
        self.w[0] = self.w_theo.w(self.x, self.ti)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)
        self.u = self.norm(self.w[0])

    def set_init_condition(self, w_0, eps=0, sigma=0):
        #  Set with given init condition
        self.w[0] = w_0(self.x)

        if eps != 0:
            self.w[0][1:-1] += np.random.uniform(-1, 1, self.N-2)*eps

        if sigma != 0:
            self.w[0] += np.exp(-sigma*self.x**2)

        self.u = self.norm(self.w[0])

    def Ah(self, n):
        g = 0.04
        i = 1j
        e = np.ones(self.N, dtype='complex_')
        e = e * (i - self.la + self.dt * self.u/2 + g * self.dt * self.u**2/2)

        if self.border.lower() == "neumann":
            e[0] += self.la/2
            e[-1] += self.la/2

        f = np.ones(self.N-1, dtype='complex_') * self.la/2

        k_ = np.array([f, e, f], dtype=object)
        offset = [-1, 0, 1]

        A = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return A

        if self.border.lower() == "periodic":
            A = A.tolil()

            A[0, -1] = self.la/2 
            A[-1, 0] = self.la/2

            A = A.tocsc()
            return A

        return A

    def Bh(self, n):
        g = 0.04
        i = 1j
        e = np.ones(self.N, dtype='complex_')
        e = e * (i + self.la - self.dt * self.u/2 - g * self.dt * self.u**2/2)

        if self.border.lower() == "neumann":
            e[0] += -self.la/2
            e[-1] += -self.la/2

        f = np.ones(self.N-1, dtype='complex_') * -self.la/2

        k_ = np.array([f, e, f], dtype=object) 
        offset = [-1, 0, 1]

        B = diags(k_, offset, format="csc", dtype='complex_')

        if self.border.lower() == "dirichlet":
            return B

        if self.border.lower() == "periodic":
            B = B.tolil()

            B[0, -1] = -self.la/2
            B[-1, 0] = -self.la/2

            B.tocsc()
            return B

        return B

    def update(self, n):
        wn = self.w[n]

        self.u = 2 * self.norm(wn) - self.u

        # Obtiene matriz A
        A = self.Ah(n)

        # Matriz B
        B = self.Bh(n)
        b = B*wn

        # Solve linear problem
        wn_sig = spsolve(A, b)

        # Update
        self.w[n+1] = wn_sig


class TSNLS_FQ(NLS_FQ_numerical):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.method = "TSSP"

    @abstractmethod
    def method_update(self, w):
        pass

    def ODE_solution(self, w):
        i = 1j
        g = 0.04
        output = np.exp((i*self.dt*(self.norm(w) + g*self.norm(w)**2))/2) * w
        return output

    def update(self, n):

        wn = self.w[n]

        w1 = self.ODE_solution(wn)

        w2 = self.method_update(w1)

        wn_sig = self.ODE_solution(w2)

        # Update
        self.w[n+1] = wn_sig


class TSSP(TSNLS_FQ):

    def __init__(self, L, N, dt, tf, pars, border, ti=0):
        super().__init__(L, N, dt, tf, pars, border, ti)
        self.method = "TSSP"

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
