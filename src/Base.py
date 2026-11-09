import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, block_array, kron, csc_matrix
from matplotlib import animation
from time import time
from abc import ABC, abstractmethod
from Utilities import *
import gzip

class NLS_solutions(ABC):
    
    def __init__(self, L, N, Nt, tf, pars, ti=0):
        i = 1j
        self.N = N
        self.L = L
        self.pars = pars
        
        self.Nt = Nt
        self.ti = ti
        self.tf = tf

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

    def __init__(self, L, N, Nt, tf, pars, border, x_0=0, L_tilde=None, ti=0, path="", save_as_files=None, save_properties=False, local=False, print_iteration=True):

        super().__init__(L, N, Nt, tf, pars, ti)
        # Execution time
        self.time = 0

        # Step sizes and total points
        self.h = np.abs(self.x[1] - self.x[0]) #2*L/self.N
        self.dt = np.abs(self.t[1] - self.t[0]) #(self.tf-self.ti)/self.Nt
        self.x_0 = x_0

        # matrices
        self.D1 = self.build_D1(self.N, self.h)
        self.D2 = self.build_D2(self.N, self.h)

        self.local = local
        self.print_iteration = print_iteration

        if L_tilde is not None:
            self.L_tilde = L_tilde

        else:
            self.L_tilde = self.L

        # Stop simulation
        self.stop = False

        # Parameters
        self.border = border
        self.save_as_files = save_as_files
        self.save_properties = save_properties
        self.path = path
        self.stop_message = "Simulation stopped due to a problem in n ="

        # background
        self.background = 0

        if self.save_properties:
            # Creates paths
            self.time_path = self.path + "time"
            self.space_path = self.path + "space"
            self.mass_path = self.path + "mass"
            self.arg_max_path = self.path + "arg_max"
            self.energy_path = self.path + "energy"
            self.norm_inf = self.path + "norm_inf"
            self.H1_norm_path = self.path + "H1_norm"
            self.H2_norm_path = self.path + "H2_norm"
            self.grad_norm_inf_path = self.path + "grad_inf"
            self.virial_path = self.path + "virial"


            # Creates arrays to save properties
            self.to_save_mass = np.zeros(self.Nt)
            self.to_save_energy = np.zeros(self.Nt)
            self.to_save_arg_max = np.zeros(self.Nt)
            self.to_save_norm_inf = np.zeros(self.Nt)
            self.to_save_H1_norm = np.zeros(self.Nt)
            self.to_save_H2_norm = np.zeros(self.Nt)
            self.to_save_grad_norm_inf = np.zeros(self.Nt)
            self.to_save_virial = np.zeros(self.Nt)

            if self.local:
                self.energy_L_path = self.path + "energy_L"
                self.energy_R_path = self.path + "energy_R"

                self.arg_max_L_path = self.path + "arg_max_L"
                self.arg_max_R_path = self.path + "arg_max_R"

                self.max_L_path = self.path + "max_L"
                self.max_R_path = self.path + "max_R"

                self.mass_L_path = self.path + "mass_L"
                self.mass_R_path = self.path + "mass_R"

                self.to_save_energy_L = np.zeros(self.Nt)
                self.to_save_energy_R = np.zeros(self.Nt)
                self.to_save_mass_L = np.zeros(self.Nt)
                self.to_save_mass_R = np.zeros(self.Nt)

                if self.background == 0:
                    self.to_save_arg_max_L = np.zeros(self.Nt)
                    self.to_save_arg_max_R = np.zeros(self.Nt)
                    self.to_save_max_L = np.zeros(self.Nt)
                    self.to_save_max_R = np.zeros(self.Nt)

        # Theo
        self.w_theo = None

        # Array to save the solution
        if self.save_as_files is None:
            self.w = np.zeros((self.Nt, self.N), dtype = 'complex')

        else:
            self.save_iter = 0


    @abstractmethod
    def update(self, n, **kwargs):
        pass

    @abstractmethod
    def energy(self, w, h, k, N):
        pass

    @abstractmethod
    def mass(self, w, h, N):
        pass
    
    def build_D1(self, N, h):
        # First derivative matrix (forward)
        D1 = diags([-1.0, 1.0], offsets=[0, 1], shape=(N-1, N), format="csr")
        D1 /= h
        return D1

    def build_D2(self, N, h):
        e = np.ones(N)

        D2 = diags(
            diagonals=[e, -2.0 * e, e],
            offsets=[-1, 0, 1],
            shape=(N, N),
            format="lil"
        )

        # Boundary adjustments for homogeneous Neumann with centered scheme
        D2[0, 1] = 2.0
        D2[-1, -2] = 2.0

        return D2.tocsr() / h**2


    def norm_h_2_plus_2(self, w, h, N):
        w0 = w[0]
        wJ = w[-1]
        return h * (np.sum(norm(w[1:N-1])) + 0.5*(norm(w0) + norm(wJ)))


    def norm_h_2_to_2(self, w, h, N):
            w0 = w[0]
            wJ = w[-1]
            return h * np.sum(norm(w[:N-1]))


    def norm_h_4_to_4(self, w, h, N):
        w0 = w[0]
        wJ = w[-1]
        return h * (np.sum(np.abs(w[1:N-1])**4) + 0.5*(np.abs(w0)**4 + np.abs(wJ)**4))


    # ------------------------------------------------------------------
    # Discrete inner products and norms (see \eqref{eq:C_inner}-\eqref{eq:norms}
    # in the theory paper). These generalize norm_h_2_plus_2, norm_h_2_to_2 and
    # norm_h_4_to_4 above (kept unchanged for backward compatibility) to the
    # three discrete products/norms used across the papers:
    #   - "dx^-" : no trapezoidal weights, drops the last spatial point
    #              (used e.g. on differences U_{j+1}-U_j, which live on J points).
    #   - "dx"   : no trapezoidal weights, keeps every point.
    #   - "dx^+" : trapezoidal weights (1/2 at both endpoints), keeps every point.
    # Method names mirror the paper's macros \ipCm, \ipC, \ipCp (complex
    # sesquilinear forms) and \ipRm, \ipR, \ipRp (their real parts).
    # ------------------------------------------------------------------

    # Complex sesquilinear inner products
    def ipCm(self, W, Z, h):
        r"""<W, Z>_{dx^-} = h * sum_{j=0}^{J-1} W_j conj(Z_j)."""
        return h * np.sum(W * np.conjugate(Z))

    def ipC(self, U, V, h):
        r"""<U, V>_{dx} = h * sum_{j=0}^{J} U_j conj(V_j) (no trapezoidal weights)."""
        return h * np.sum(U * np.conjugate(V))

    def ipCp(self, U, V, h):
        r"""<U, V>_{dx^+} = h * (U_0 conj(V_0)/2 + sum_{j=1}^{J-1} U_j conj(V_j) + U_J conj(V_J)/2)."""
        return h * (0.5 * U[0] * np.conjugate(V[0])
                     + np.sum(U[1:-1] * np.conjugate(V[1:-1]))
                     + 0.5 * U[-1] * np.conjugate(V[-1]))

    # Real inner products: real part of the sesquilinear forms above
    def ipRm(self, W, Z, h):
        r"""Re <W, Z>_{dx^-}."""
        return np.real(self.ipCm(W, Z, h))

    def ipR(self, U, V, h):
        r"""Re <U, V>_{dx}."""
        return np.real(self.ipC(U, V, h))

    def ipRp(self, U, V, h):
        r"""Re <U, V>_{dx^+}."""
        return np.real(self.ipCp(U, V, h))

    # Norms to the p-th power (consistent with norm_h_2_plus_2/norm_h_4_to_4 above,
    # which return the norm raised to p rather than the norm itself)
    def norm_dx_minus(self, W, h, p=2):
        r"""||W||_{p, dx^-}^p = h * sum_{j=0}^{J-1} |W_j|^p."""
        return h * np.sum(np.abs(W)**p)

    def norm_dx(self, U, h, p=2):
        r"""||U||_{p, dx}^p = h * sum_{j=0}^{J} |U_j|^p (no trapezoidal weights)."""
        return h * np.sum(np.abs(U)**p)

    def norm_dx_plus(self, U, h, p=2):
        r"""||U||_{p, dx^+}^p = h * (|U_0|^p/2 + sum_{j=1}^{J-1} |U_j|^p + |U_J|^p/2)."""
        return h * (0.5 * np.abs(U[0])**p
                     + np.sum(np.abs(U[1:-1])**p)
                     + 0.5 * np.abs(U[-1])**p)

    def norm_H1_dx(self, U, D1=None, h=None):
        r"""Discrete H^1_dx norm, as defined in the theory paper:
        ||U||_{H1_dx}^2 = ||U||_{2, dx^+}^2 + ||Dxp U||_{2, dx^-}^2.
        U is a full-length (N-point) array; Dxp U = D1 @ U has N-1 points and
        is compared with norm_dx_minus, which is exactly the ||.||_{2,dx^-}
        convention (no dropped/re-weighted points needed since D1 @ U already
        has J = N-1 entries). Defaults to this instance's own D1 and h, but
        both can be overridden (e.g. to reuse a different grid's operator).
        """
        if D1 is None:
            D1 = self.D1
        if h is None:
            h = self.h
        return np.sqrt(self.norm_dx_plus(U, h, p=2) + self.norm_dx_minus(D1 @ U, h, p=2))


    def get_mass(self, plot=False, name=None, **kwargs):

        if self.save_as_files is not None and self.save_properties:
            if self.local:
                wn_r = self.w[np.abs(self.x-self.x_0)<self.L_tilde]
                m_r = self.mass(wn_r, self.h, self.N)
                #m_r = self.h * np.sum(norm(wn_r))
    
                self.to_save_mass_R[self.save_iter] = m_r

                wn_l = self.w[np.abs(self.x+self.x_0)<self.L_tilde]
                m_l = self.mass(wn_l, self.h, self.N)
                #m_l = self.h * np.sum(norm(wn_l))
    
                self.to_save_mass_L[self.save_iter] = m_l

            wn = self.w
            m = self.mass(wn, self.h, self.N)
            #m = self.h * np.sum(norm(wn))
    
            self.to_save_mass[self.save_iter] = m

        else:
            mass = np.zeros(self.Nt)
            for n in range(self.Nt):
                wn = self.w[n]
                m = self.mass(wn, self.h, self.N)
                #m = self.h * np.sum(norm(self.w[n]))

                mass[n] = m
                
            if plot:
                plt.plot(self.t, mass-mass[0], **kwargs)
    
                plt.xlabel(r"$t$")
                plt.ylabel(r"$N(t)-N(0)$")
                plt.tight_layout()
    
                if name is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
    
            return mass
    

    def get_energy(self, w=None, plot=False, name=None, **kwargs):
        k = self.pars["k"]

        if w is not None:
            e = self.energy(w, self.h, k, self.N)
            return e

        if self.save_as_files is not None and self.save_properties:
            if self.local:
                wn_r = self.w[np.abs(self.x-self.x_0)<self.L_tilde]

                e_r = self.energy(wn_r, self.h, k, self.N)

                wn_l = self.w[np.abs(self.x+self.x_0)<self.L_tilde]

                e_l = self.energy(wn_l, self.h, k, self.N)

                self.to_save_energy_R[self.save_iter] = e_r
                self.to_save_energy_L[self.save_iter] = e_l

            wn = self.w

            e = self.energy(wn, self.h, k, self.N)

            self.to_save_energy[self.save_iter] = e

        else:
            energy = np.zeros(self.Nt)
            for n in range(self.Nt):
                wn = self.w[n]
                e = self.energy(wn, self.h, k, self.N)
                energy[n] = e

            if plot:
                energy = np.trim_zeros(energy)
                plt.plot(self.t, energy-energy[0], **kwargs)
        
                plt.xlabel(r"$t$")
                plt.ylabel(r"$E(t)-E(0)$")
                plt.tight_layout()
        
                if name is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
        
            return energy


    def set_w_theo(self, w_theo, eps=0, sigma=0):
        self.w_theo = w_theo

        if self.save_as_files is None:
            #  Set init condition with theo class
            self.w[0] = self.w_theo(self.x, self.ti)
            if eps != 0:
                self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
            if sigma != 0:
                self.w[0] += np.exp(-sigma*self.x**2)

        else:
            self.w = self.w_theo(self.x, self.ti)
            save_file(self.w, self.path+self.save_as_files+f"{self.save_iter:010d}")
            if self.save_properties:

                self.to_save_error_L2 = np.zeros(self.Nt)
                self.to_save_error_H1 = np.zeros(self.Nt)
                self.to_save_error_inf = np.zeros(self.Nt)
                self.to_save_error_grad_inf = np.zeros(self.Nt)

                self.error_L2_path = self.path + "error_L2"
                self.error_H1_path = self.path + "error_H1"
                self.error_inf_path = self.path + "error_inf"
                self.error_grad_inf_path = self.path + "error_grad_inf"    

                # Save to files
                self.save_properties_func()

                self.get_error()
                self.get_mass()
                self.get_energy()
                self.get_arg()
                self.get_norm_inf()
                self.get_H_norms()
                self.get_virial()

            self.save_iter += 1


    def set_init_condition(self, w_0, eps=0, sigma=0):

        if self.save_as_files is None:
            #  Set with given init condition
            self.w[0] = w_0(self.x)
            if eps != 0:
                self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
            if sigma != 0:
                self.w[0] += np.exp(-sigma*self.x**2)

        else:
            self.w = w_0(self.x)
            save_file(self.w, self.path+self.save_as_files+f"{self.save_iter:010d}")
            if self.save_properties:
                if self.w_theo is not None:
                    self.to_save_error_L2 = np.zeros(self.Nt)
                    self.to_save_error_H1 = np.zeros(self.Nt)
                    self.to_save_error_inf = np.zeros(self.Nt)
                    self.to_save_error_grad_inf = np.zeros(self.Nt)
    
                    self.error_L2_path = self.path + "error_L2"
                    self.error_H1_path = self.path + "error_H1"
                    self.error_inf_path = self.path + "error_inf"
                    self.error_grad_inf_path = self.path + "error_grad_inf"
        
                    # Save to files
                    self.save_properties_func()
                    self.get_error()

                else:
                    # Save to files
                    self.save_properties_func()

                self.get_mass()
                self.get_energy()
                self.get_arg()
                self.get_norm_inf()
                self.get_H_norms()
                self.get_virial()

            self.save_iter += 1
            
    
    def set_init_array(self, w_0, eps=0, sigma=0):

        if self.save_as_files is None:
            #  Set with given init array
            self.w[0] = w_0
            if eps != 0:
                self.w[0][1: -1] += np.random.uniform(-1, 1, self.N-2)*eps
            if sigma != 0:
                self.w[0] += np.exp(-sigma*self.x**2)

        else:
            self.w = w_0
            save_file(self.w, self.path+self.save_as_files+f"{self.save_iter:010d}")
            if self.save_properties:

                if self.w_theo is not None:
                    self.to_save_error_L2 = np.zeros(self.Nt)
                    self.to_save_error_H1 = np.zeros(self.Nt)
                    self.to_save_error_inf = np.zeros(self.Nt)
                    self.to_save_error_grad_inf = np.zeros(self.Nt)
    
                    self.error_L2_path = self.path + "error_L2"
                    self.error_H1_path = self.path + "error_H1"
                    self.error_inf_path = self.path + "error_inf"
                    self.error_grad_inf_path = self.path + "error_grad_inf"
        
                    # Save to files
                    self.save_properties_func()
                    self.get_error()

                else:
                    # Save to files
                    self.save_properties_func()

                self.get_mass()
                self.get_energy()
                self.get_arg()
                self.get_norm_inf()
                self.get_H_norms()
                self.get_virial()

            self.save_iter += 1

    def solve(self, Nelem=None, b_tol=None, return_max=False, print_tf=False, energy_perc=None, norm_inf_stop=None, **kwargs):
        init_time = time()
        self.time = init_time

        if type(Nelem) is np.ndarray:
            self.save_index = 0
        
        if self.save_as_files is None:

            if energy_perc is not None:
                energy_0 = self.get_energy(self.w[0])

            for n in range(self.Nt-1):
                if self.stop:
                    print(self.stop_message, n)
                    break

                self.w[n+1] = self.update(n, **kwargs)

                if b_tol is not None:
                    border_cond_left = np.abs(norm(self.w[n+1][0])-self.background) > b_tol
                    border_cond_right = np.abs(norm(self.w[n+1][-1])-self.background) > b_tol
                    if border_cond_left or border_cond_right:
                        self.stop = True
                        self.stop_message = "Simulation reached the border in n ="

                if energy_perc is not None:
                    actual_energy = self.get_energy(self.w[n+1])
                    variation_perc = (actual_energy - energy_0)/energy_0 * 100

                    if np.abs(variation_perc) >= energy_perc:
                        self.stop = True
                        self.stop_message = "Energy varied {}% in n =".format(variation_perc)

                if norm_inf_stop is not None:
                    if self.background == 0:
                        actual_norm_inf = np_norm(self.w[n+1], np.inf)
                    else:
                        actual_norm_inf = np.min(np.abs(self.w[n+1]))

                    if actual_norm_inf > norm_inf_stop:
                        self.stop = True
                        self.stop_message = "norm_inf exceeded threshold ({}) in n =".format(norm_inf_stop)

        else:
            if return_max:
                w_max = np.zeros(self.Nt)
                w_max[0] = np.max(np.abs(self.w))

            if energy_perc is not None:
                energy_0 = self.get_energy(self.w)

            for n in range(self.Nt-1):

                new_w = self.update(n, **kwargs)
                
                if b_tol is not None:
                    border_cond_left = np.abs(norm(self.w[0])-self.background) > b_tol
                    border_cond_right = np.abs(norm(self.w[-1])-self.background) > b_tol
                    if border_cond_left or border_cond_right:
                        self.stop = True
                        self.stop_message = "Simulation reached the border in n ="

                if energy_perc is not None:
                    actual_energy = self.get_energy(new_w)
                    variation_perc = (actual_energy - energy_0)/energy_0 * 100

                    if variation_perc >= energy_perc:
                        self.stop = True
                        self.stop_message = "Energy increased {}% in n =".format(variation_perc)

                if norm_inf_stop is not None:
                    if self.background == 0:
                        actual_norm_inf = np_norm(new_w, np.inf)
                    else:
                        actual_norm_inf = np.min(np.abs(new_w))

                    if actual_norm_inf > norm_inf_stop:
                        self.stop = True
                        self.stop_message = "norm_inf exceeded threshold ({}) in n =".format(norm_inf_stop)
                        new_w = self.w

                self.w = new_w

                if return_max:
                    w_max[n+1] = np.max(np.abs(self.w))

                if print_tf:
                    print("Actual final time: ", self.tf)

                if Nelem is None:
                    save_file(self.w, self.path+self.save_as_files+f"{n:010d}")

                elif type(Nelem) is int and (n+2)%Nelem==0:
                    save_file(self.w, self.path+self.save_as_files+f"{(n+2):010d}")

                elif type(Nelem) is float:
                    print("Warning: Nelem is not integer. No file will be saved.")

                elif type(Nelem) is np.ndarray:
                    print("Next element to save: ", Nelem[self.save_index])
                    if (n+2)==Nelem[self.save_index]:
                        self.save_index += 1
                        save_file(self.w, self.path+self.save_as_files+f"{(n+2):010d}")

                if self.save_properties:
                    if self.w_theo is not None:
                        error_L2, error_inf, error_H1, error_grad_inf = self.get_error()
                        print("L2 error: ", error_L2, " Linf error: ", error_inf)

                    self.get_mass()
                    self.get_energy()
                    self.get_arg()
                    self.get_norm_inf()
                    self.get_H_norms()
                    self.get_virial()

                    self.save_properties_func()
        
                self.save_iter += 1

                if self.stop:
                    save_file(self.w, self.path+self.save_as_files+f"{(n+2):010d}")
                    print(self.stop_message, n)
                    break

        final_time = time()
        self.time = final_time - init_time

        if return_max:
            return np.array(w_max)

    def get_error(self, t=None, p=2):
        D1 = self.build_D1(self.N, self.h)
        if self.save_as_files is not None and self.save_properties and t is None:
            w_sim = self.w
            t = self.t[self.save_iter]#(self.save_iter - 1) * self.Nt
            w_theo = self.w_theo(self.x, t)

            err_L2 = np.sqrt(self.h) * np_norm((w_sim - w_theo), 2)
            self.to_save_error_L2[self.save_iter] = err_L2

            err_inf = np_norm((w_sim - w_theo), np.inf)
            self.to_save_error_inf[self.save_iter] = err_inf

            err_H1 = self.norm_H1_dx(w_sim - w_theo, D1=D1)
            self.to_save_error_H1[self.save_iter] = err_H1

            err_grad_inf = np_norm((D1 @ (w_sim - w_theo)), np.inf)
            self.to_save_error_grad_inf[self.save_iter] = err_grad_inf

            return err_L2, err_inf, err_H1, err_grad_inf

        elif t is None:
            error = np.zeros(self.Nt)

            for n in range(self.Nt):
                t = self.t[n]
                if p==2:
                    diff = norm(self.w[n] - self.w_theo(self.x, t))
                    sum_ = np.sum(diff)
                    e = np.sqrt(self.h * sum_)
                    error[n] = e
                if p==np.inf:
                    e = self.h * np_norm(self.w[n] - self.w_theo(self.x, t), p)
                    error[n] = e

                if p=="H1":
                    diff = self.w[n] - self.w_theo(self.x, t)
                    e = self.norm_H1_dx(diff)
                    error[n] = e

                if p=="grad_inf":
                    diff = self.w[n] - self.w_theo(self.x, t)
                    e = np_norm(self.D1 @ diff, np.inf)
                    error[n] = e
            return error

        else:
            n = int((t - self.ti)/self.dt)
            w_sim = self.w[n]
            w_theo = self.w_theo(self.x, t)
            
            diff = norm(w_sim - w_theo)
            sum_ = np.sum(diff)
            output = np.sqrt(self.h * sum_)
            return output

    def get_norm_inf(self, plot=False, name=None, path=None):

        if self.save_as_files is not None and self.save_properties:

            if self.local:
                cond_l = np.abs(self.x+self.x_0)<self.L_tilde
                wn_l = self.w[cond_l]

                cond_r = np.abs(self.x-self.x_0)<self.L_tilde
                wn_r = self.w[cond_r]

                if self.background==0:
                    max_L = np_norm(wn_l, np.inf)
                    self.to_save_max_L[self.save_iter] = max_L

                    max_R = np_norm(wn_r, np.inf)
                    self.to_save_max_R[self.save_iter] = max_R

                if self.background==1:
                    min_L = np.min(np.abs(wn_l))
                    self.to_save_min_L[self.save_iter] = min_L

                    min_R = np.min(np.abs(wn_r))
                    self.to_save_min_R[self.save_iter] = min_R
    
            wn = self.w

            if self.background==0:
                norm_inf = np_norm(wn, np.inf)

            if self.background==1:
                norm_inf = np.min(np.abs(wn))
            
            self.to_save_norm_inf[self.save_iter] = norm_inf

        else:
            norm_inf = np.zeros(self.Nt)
            for n in range(self.Nt):

                if self.background==0:
                    n_inf = np_norm(self.w[n], np.inf)

                if self.background==1:
                    n_inf = np.min(np.abs(self.w[n]))

                norm_inf[n] = n_inf
                
            if plot:
                plt.plot(self.t, norm_inf, ".")
    
                plt.xlabel(r"$t$")
                if self.background==0:
                    plt.ylabel(r"Max$|\Psi(t)|$")

                if self.background==1:
                    plt.ylabel(r"Min$|\Psi(t)|$")
                plt.tight_layout()
    
                if path is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
    
            return norm_inf

    def get_arg(self):

        if self.save_as_files is not None and self.save_properties:

            if self.local:
                cond_l = np.abs(self.x+self.x_0)<self.L_tilde
                wn_l = self.w[cond_l]

                cond_r = np.abs(self.x-self.x_0)<self.L_tilde
                wn_r = self.w[cond_r]

                x_l = self.x[cond_l]
                x_r = self.x[cond_r]

                if self.background == 0:
                    arg_max_L = x_l[np.argmax(norm(wn_l))]
                    self.to_save_arg_max_L[self.save_iter] = arg_max_L
                    
                    arg_max_R = x_r[np.argmax(norm(wn_r))]
                    self.to_save_arg_max_R[self.save_iter] = arg_max_R

                if self.background == 1:
                    arg_min_L = x_l[np.argmin(norm(wn_l))]
                    self.to_save_arg_min_L[self.save_iter] = arg_min_L
                    
                    arg_min_R = x_r[np.argmin(norm(wn_r))]
                    self.to_save_arg_min_R[self.save_iter] = arg_min_R

            wn = self.w
            if self.background == 0:
                arg_max = self.x[np.argmax(norm(wn))]
                self.to_save_arg_max[self.save_iter] = arg_max

            if self.background == 1:
                arg_min = self.x[np.argmin(norm(wn))]
                self.to_save_arg_min[self.save_iter] = arg_min


    def get_H_norms(self, plot=False, name=None, path=None):

        if self.save_as_files is not None and self.save_properties:
            wn = self.w
            # Seminorms as defined for the blow-up experiments in the paper:
            # |Psi|_{H1_dx} = ||Dxp Psi||_{2,dx^-} and |Psi|_{H2_dx} = ||Dxx Psi||_{2,dx}
            H1_norm = np.sqrt(self.norm_dx_minus(self.D1 @ wn, self.h, p=2))
            H2_norm = np.sqrt(self.norm_dx(self.D2 @ wn, self.h, p=2))
            grad_norm_inf = np_norm(self.D1 @ wn, np.inf)

            self.to_save_H1_norm[self.save_iter] = H1_norm
            self.to_save_H2_norm[self.save_iter] = H2_norm
            self.to_save_grad_norm_inf[self.save_iter] = grad_norm_inf

        else:
            grad_norm = np.zeros(self.Nt)
            for n in range(self.Nt):
                g_n = np.sqrt(self.norm_dx_minus(self.D1 @ self.w[n], self.h, p=2))
                grad_norm[n] = g_n
                
            if plot:
                plt.plot(self.t, grad_norm, ".")
    
                plt.xlabel(r"$t$")
                plt.ylabel(r"$|\Delta \Psi|_{L^2}(t)$")
                plt.tight_layout()
    
                if path is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
    
            return grad_norm

    def get_virial(self, plot=False, name=None, path=None):

        if self.save_as_files is not None and self.save_properties:
            wn = self.w
            virial = self.h * np_norm(self.x*wn, 2)**2

            self.to_save_virial[self.save_iter] = virial

        else:
            virial = np.zeros(self.Nt)
            for n in range(self.Nt):
                v_n = self.h * np_norm(self.x*self.w[n], 2)**2
                virial[n] = v_n

            if plot:
                plt.plot(self.t, virial, ".")
    
                plt.xlabel(r"$t$")
                plt.ylabel(r"$V(t)$")
                plt.tight_layout()
    
                if path is not None:
                    plt.savefig(self.path+name, dpi=300, bbox_inches = "tight")
                plt.show()
    
            return virial


    def get_execution_time(self):
        return self.time


    def plot(self, t,  **kwargs):
        n = int((t - self.ti)/self.dt)
        plt.plot(self.x, norm(self.w[n]),  **kwargs)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$|\Psi|^2$")


    def plot_real(self, t,  **kwargs):
        n = int((t - self.ti)/self.dt)
        plt.plot(self.x, np.real(self.w[n]),  **kwargs)
        plt.xlabel(r"$x$")
        plt.ylabel(r"Re$(\Psi)$")

    def plot_imag(self, t, **kwargs):
        n = int((t - self.ti)/self.dt)
        plt.plot(self.x, np.imag(self.w[n]), **kwargs)
        plt.xlabel(r"$x$")
        plt.ylabel(r"Im$(\Psi)$")


    def animate_solution(self, name, N_frames, theo=False, coord="norm", auto_adjust=True, N_final=False):
        # Create figure
        fig, ax = plt.subplots()

        # For empty plot
        empty_line = np.full(len(self.x), fill_value=None)

        # Plot init instant
        if coord.lower() == "norm":
            line1, = ax.plot(self.x, norm(self.w[0]), "-", label="Simulation")
        
        elif coord.lower() == "real":
            line1, = ax.plot(self.x, np.real(self.w[0]), "-", label="Simulation")
        
        elif coord.lower() == "imag":
            line1, = ax.plot(self.x, np.imag(self.w[0]), "-", label="Simulation")


        line2, = ax.plot(self.x, empty_line, "--", label="Theo", alpha=.9)
        # Plotting with theo
        if theo and coord.lower() == "norm":
            line2.set_ydata(norm(self.w_theo(self.x, self.ti)))
                
        elif theo and coord.lower() == "real":
            line2.set_ydata(np.real(self.w_theo(self.x, self.ti)))
            
        elif theo and coord.lower() == "imag":
            line2.set_ydata(np.imag(self.w_theo(self.x, self.ti)))

        ax.set_xlabel(r"$x$")
        if coord.lower() == "norm":
            ax.set_ylabel(r"$|\Psi|^2$")

        if coord.lower() == "real":
            ax.set_ylabel(r"$Re(\Psi)$")

        if coord.lower() == "imag":
            ax.set_ylabel(r"$Im(\Psi)$")

        if coord.lower() == "norm":
            if not auto_adjust:
                max_val = norm(self.w).max().max()
                min_val = norm(self.w).min().min()
                ax.set_ylim(min_val*0.9, max_val*1.1)

        if coord.lower() == "real":
            if not auto_adjust:
                max_val = np.real(self.w).max().max()
                min_val = np.real(self.w).min().min()
                ax.set_ylim(min_val*0.9, max_val*1.1)

        if coord.lower() == "imag":
            if not auto_adjust:
                max_val = np.imag(self.w).max().max()
                min_val = np.imag(self.w).min().min()
                ax.set_ylim(min_val*0.9, max_val*1.1)

        fig.set_tight_layout(True)

        # Animate function
        def animate(j, theo):

            # Update plot
            
            if coord.lower() == "norm":
                if auto_adjust:
                    ax.set_ylim(min(norm(self.w[j]))-.2, max(norm(self.w[j]))+.2)
                #ax.set_ylim(-0.1, 17)
                line1.set_data(self.x, norm(self.w[j]))
                
            if coord.lower() == "real":
                if auto_adjust:
                    ax.set_ylim(min(np.real(self.w[j]))-.2, max(np.real(self.w[j]))+.2)
                line1.set_data(self.x, np.real(self.w[j]))
                
            if coord.lower() == "imag":
                if auto_adjust:
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
        if N_final:
            N_final = len(np.trim_zeros(self.w))

        else:
            N_final = self.Nt
    
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, N_final, N_frames), 
                                      interval=150, fargs=[theo])

        ani.save(self.path+name, bitrate=-1, codec="libx264")
        plt.show()
        return ani
    
    def get_colormap(self, name=None):
        fig, ax = plt.subplots(1, 1)
        im = ax.pcolorfast(self.x, self.t, norm(self.w))
        
        fig.colorbar(im)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        
        if name is not None:
            fig.savefig(self.path+name)
        plt.show()

    # Function to save the solution as a file
    def savefile(self, name, NumElem=None):
        
        if NumElem is None:
            with gzip.open(self.path+name, 'wb') as f:
                np.save(f, self.w)

        else:
            idx = np.round(np.linspace(0, self.Nt, NumElem)).astype(int)
            with gzip.open(self.path+name, 'wb') as f:
                np.save(f, self.w[idx])


    # Function to save properties of the solution
    def save_properties_func(self):

        save_file(self.t, self.time_path)
        save_file(self.x, self.space_path)
        save_file(self.to_save_mass, self.mass_path)
        save_file(self.to_save_energy, self.energy_path)
        save_file(self.to_save_norm_inf, self.norm_inf)
        save_file(self.to_save_H1_norm, self.H1_norm_path)
        save_file(self.to_save_H2_norm, self.H2_norm_path)
        save_file(self.to_save_grad_norm_inf, self.grad_norm_inf_path)
        save_file(self.to_save_virial, self.virial_path)

        if self.background == 0:
            save_file(self.to_save_arg_max, self.arg_max_path)

        if self.background == 1:
             save_file(self.to_save_arg_min, self.arg_min_path)

        if self.local:
            save_file(self.to_save_mass_L, self.mass_L_path)
            save_file(self.to_save_mass_R, self.mass_R_path)
            
            save_file(self.to_save_energy_L, self.energy_L_path)
            save_file(self.to_save_energy_R, self.energy_R_path)

            if self.background == 0:
                save_file(self.to_save_arg_max_L, self.arg_max_L_path)
                save_file(self.to_save_arg_max_R, self.arg_max_R_path)

                save_file(self.to_save_max_L, self.max_L_path)
                save_file(self.to_save_max_R, self.max_R_path)

            if self.background == 1:
                save_file(self.to_save_arg_min_L, self.arg_min_L_path)
                save_file(self.to_save_arg_min_R, self.arg_min_R_path)

                save_file(self.to_save_min_L, self.min_L_path)
                save_file(self.to_save_min_R, self.min_R_path)

        if self.w_theo is not None:
            save_file(self.to_save_error_L2, self.error_L2_path)
            save_file(self.to_save_error_H1, self.error_H1_path)
            save_file(self.to_save_error_inf, self.error_inf_path)
            save_file(self.to_save_error_grad_inf, self.error_grad_inf_path)
