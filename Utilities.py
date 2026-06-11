import numpy as np
import os
from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, csc_matrix
import gzip
from matplotlib import animation


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Norm function
def norm(w):
    return np.abs(w)**2

# Sech function
def sech(x):
    return 1/np.cosh(x)

# Newton method
#def newton(x_0, f, f_prime, h, tol=1e-10, max_iter=1000):
#    x_k = x_0
#    error = np.add.reduce(np.abs(f(x_k)))
#    j = 0
#    while error > tol:
#        #print(error)
#        f_1 = -f(x_k)
#        f_2 = f_prime(x_k)
        #plt.plot(norm(f_1[:len(x_k)//4]))
        #plt.plot(norm(x_k[:len(x_k)//2]))
        #plt.show()
#        delta_x = spsolve(f_2, f_1, permc_spec="MMD_AT_PLUS_A")
#        x_k_1 = x_k + delta_x
#        error = np_norm(f(x_k_1), 2) #np.sqrt(h) * np_norm(delta_x, 2) #
#        x_k = x_k_1
#        j += 1

#        if j>max_iter:
#            print("Simulation reaches maximum iterations. Stoping...")
#            error = float('NaN')
#            break

#    return x_k, error, j


def l2h(v, h):
    return np.sqrt(h) * np_norm(v.ravel(), 2)


# Newton method
def newton(x_0, f, f_prime, h,
           tolF=1e-10,
           tolX_abs=0.0, tolX_rel=0, #0,
           max_iter=1000):
    x_k = x_0
    r = f(x_k)
    r0_norm = l2h(r, h)
    tolF_eff = max(tolF, tolF * r0_norm)  # relative criteria for the residue

    j = 0

    if tolX_abs is None:
        tolX_abs = 0

    if tolX_rel is None:
        tolX_rel = 0

    while True:
        # principal criteria
        r_norm = l2h(r, h)
        if r_norm <= tolF_eff:
            break

        # Newton step
        J = f_prime(x_k)
        delta_x = spsolve(J, -r, permc_spec="MMD_AT_PLUS_A")

        # Secondary criteria: step size
        step_norm = l2h(delta_x, h)
        if step_norm <= (tolX_abs + tolX_rel * l2h(x_k, h)):
            break

        # Update
        x_k = x_k + delta_x
        r = f(x_k)

        j += 1
        if j > max_iter:
            print("Simulation reaches maximum iterations. Stopping...")
            r_norm = float('NaN')
            break
    return x_k, r_norm, j


# Newton method
#def newton(x_0, f, f_prime, h,
#           tolF=1e-10,
#           tolX_abs=0.0, tolX_rel=0, #0,
#           max_iter=1000):
#    x_k = x_0
#    r = f(x_k)
#    r0_norm = l2h(r, h)
#    tolF_eff = max(tolF, tolF * r0_norm)  # relative criteria for the residue

#    j = 0

#    if tolX_abs is None:
#        tolX_abs = 0

#    stopping_criteria = "Principal"
#    principal_iter = None
#    secondary_iter = None

#    while True:
        # principal criteria
#        r_norm = l2h(r, h)
#        principal_met = (r_norm <= tolF_eff)
#        if principal_met and principal_iter is None:
#            principal_iter = j

        # Newton step
#        J = f_prime(x_k)
#        delta_x = spsolve(J, -r, permc_spec="MMD_AT_PLUS_A")

        # Secondary criteria: step size
#        step_norm = l2h(delta_x, h)
#        secondary_met = (step_norm <= (tolX_abs + tolX_rel * l2h(x_k, h)))
#        if secondary_met and secondary_iter is None:
#            secondary_iter = j

#        if principal_met and secondary_met:
#            if principal_iter > secondary_iter:
#                stopping_criteria = "Principal"
#            elif secondary_iter > principal_iter:
#                stopping_criteria = "Secondary"
#            else:
#                stopping_criteria = "Both"
#            break

        # Update
#        x_k = x_k + delta_x
#        r = f(x_k)

#        j += 1
#        if j > max_iter:
#            print("Simulation reaches maximum iterations. Stopping...")
#            r_norm = float('NaN')
#            break

#    print("Stopping criteria: ", stopping_criteria)
#    return x_k, r_norm, j



# Fixed point
def fixed_point(x_0, g, h, tol=1e-12, max_iter=1000):
    x_k = x_0
    j = 0
    error = 1
    while error > tol:
        #plt.plot(norm(x_k), norm(g(x_k)))
        #plt.plot(norm(x_k))
        #plt.show()
        #print("Infty norm of g: ", np_norm(g(x_k), np.infty))
        
        x_k_1 = g(x_k)
        error = np_norm((x_k - x_k_1), 2)

        x_k = x_k_1
        j += 1

        if j > max_iter:
            print("Simulation reaches maximum iterations. Stoping...")
            error = float('NaN')
            break

    return x_k_1, error, j

# Save a file
def save_file(w, path):
    path = path + ".npy.gz"
    with gzip.open(path, 'wb') as f:
        np.save(f, w)


# One dimensional Newton
def newton_1D(x_0, f, f_prime, tol=1e-8):
    x_k = x_0
    error = np.add.reduce(np.abs(f(x_k)))
    while error > tol:
        f_1 = -f(x_k)
        f_2 = f_prime(x_k)
        delta_x = f_1/f_2
        x_k_1 = x_k + delta_x
        error = np_norm([x_k - x_k_1], 2)
        x_k = x_k_1

    if error != error:
        return 0, 1

    return x_k, error

#def p(c, k):

    
#P'(c)
def P_p(c, k):
    sqrt = np.sqrt((1 - 2 * k)/(2 - c**2))
    frac = (3 * c**2 * k - 4 * k - 1)/(4 * np.sqrt(-k))
    
    sum_1 = frac * np.arctanh(np.sqrt(-k)/sqrt)
    sum_2 = 3 * (2 - c**2)/4 * sqrt
    output = sum_1 - sum_2
    return output


#P''(c)
def P_pp(c, k):
    frac = 6 * c * k/(4 * np.sqrt(-k))
    sqrt = np.sqrt((1 - 2 * k)/(2 - c**2))
    sum_1 = frac * np.arctanh(np.sqrt(-k)/sqrt)
    
    frac_2 = (3 * c**2 * k - 4 * k - 1)/4
    sum_2 = frac_2 * sqrt * c/(1 - k * c**2) + 3/4 * c * sqrt
    output = sum_1 - sum_2
    return output

def build_D1(N, h):
    # First derivative matrix
    D1 = diags([-1.0, 1.0], offsets=[0, 1], shape=(N-1, N), format="csr")/h
    return D1

# Energy from files
def get_energy(w, h, k):

    N = len(w[0])
    Nt = len(w)

    energy = np.zeros(Nt)
    D1 = build_D1(N, h)

    for n in range(Nt):
        wn = w[n]
        e = h * np.sum(1/2 * norm(D1 @ wn) + 
                       1/4 * (1 - (norm(wn)))**2 - 
                       k/4 * (D1 @ norm(wn))**2)
        energy[n] = e
        
    return energy

# Errors from files
def get_errors(w, w_theo, tf, h, x, p):
    N = len(w[0])
    Nt = len(w)
    dt = tf/Nt

    error = np.zeros(Nt)

    for n in range(Nt):
        wn = w[n]
        t = n * dt
        e = np.sqrt(h) * np_norm((wn - w_theo(x, t)), p)
        error[n] = e
        
    return error

# Test energy is well calculated
def A(c, k):
    return 3*c**4*k**2 - 8*c**2*k**2 - 2*c**2*k + 8*k - 1

def B(c, k):
    return 3*c**2*k - 4*k - 1

def C(c, k):
    return c**2*k - 4*k - 1

def D(c, k):
    return np.sqrt((1 - 2*k) * (2 - c**2))

def L_c(c, k):
    return np.sqrt((2 - c**2) / (1 - 2*k))

s = 1
def E_D_1_3(c, k, s):
    return (A(c, k) * np.arctan(np.sqrt(k) * L_c(c, k)) - s * np.sqrt(k) * B(c, k) * D(c, k)) / (16 * k**(3/2))

def E_D_2(c, k, s):
    return -(A(c, k) * np.arctanh(np.sqrt(np.abs(k)) * L_c(c, k))- np.sqrt(np.abs(k)) * B(c, k) * D(c, k)) / (16 * np.abs(k)**(3/2))

# Exponential linspace
def my_lin(lb, ub, steps, spacing):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)], dtype=int)


# Creates a folder
def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

# Scientific notation formatter
def sci_notation(number:float, n_decimals: int=2) -> str:
    sci_notation = f"{number:e}"
    decimal_part, exponent = sci_notation.split("e")
    decimal_part = float(decimal_part)
    exponent = int(exponent)

    if decimal_part == 1.0:
        return r" $1 \times 10^{" +str(exponent)+"}$"

    else:
        return fr"{decimal_part:.{n_decimals}f}"+r" $ \times 10^{" +str(exponent)+"}$"


# 3D line plot
def line_plot_3D(w, x, t, path, labels, Nt_plot=35, N_plot=200, L=None, tf=None, x_rot=-80, z_rot=30, z_pad=-5, y_pad=-12, z_tick_pad=-1, sci=False):

    fig = plt.figure(layout="constrained")
    ax = fig.add_axes(111, projection="3d")   
    ax.grid(False)
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    if L is None:
        L = x[-1]

    x_to_plot = x[np.abs(x)<L]

    if tf is None:
        tf = t[-1]

    t_to_plot = t[t<tf]
    
    Nt = len(t_to_plot)
    ind_to_plot = np.linspace(0, Nt-1, Nt_plot, dtype=int)
    xs_to_plot = np.linspace(0, len(x_to_plot)-1, N_plot, dtype=int)
    X, T = np.meshgrid(x_to_plot[xs_to_plot], t_to_plot[ind_to_plot])

    w_to_plot_time = w[t<tf]
    w_to_plot = w_to_plot_time[:, np.abs(x)<L]
    w_ind_time = w_to_plot[ind_to_plot]
    w_to_plot = w_ind_time[:, xs_to_plot]
    
    ax.plot_surface(X, T, norm(w_to_plot), rstride=1, 
                    cstride=len(xs_to_plot), linewidth=.3, 
                    color="white", edgecolor='#0c5da5', shade=False)
    
    ax.set_xlabel(labels[0], fontsize=7)
    ax.set_ylabel(labels[1], fontsize=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(labels[2], fontsize=7)
    
    ax.tick_params(axis="y", pad=-3)
    ax.tick_params(axis="z", pad=z_tick_pad)
    ax.tick_params(axis="x", pad=-5)
    
    ax.xaxis.labelpad=-12
    ax.yaxis.labelpad= y_pad
    ax.zaxis.labelpad= z_pad
    
    ax.tick_params(labelsize=6)
    ax.view_init(z_rot, x_rot)
    
    ax.set_box_aspect((4, 4, 4))

    if L is None:
        L = x[-1]
    ax.set_xlim(-L, L)

    if tf is None:
        tf = t[-1]
    ax.set_ylim(0, tf)
    
    ax.xaxis.line.set_lw(0.5)
    ax.yaxis.line.set_lw(0.5)
    ax.zaxis.line.set_lw(0.5)

    if sci:
        plt.ticklabel_format(style='sci',scilimits=(-3,4),axis="y")
    
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_snapshots(x, t, w, times_to_plot, figsize=(6, 7), L=None, save_path=None, text_pos=(0.04, 0.72)):
    nplots = len(times_to_plot)
    fig, axes = plt.subplots(nplots, 1, figsize=figsize, sharex=True)

    if nplots == 1:
        axes = [axes]

    if L is not None:
        for ax in axes:
            ax.set_xlim(-L, L)

    for i, (ax, tsel) in enumerate(zip(axes, times_to_plot)):
        k = np.argmin(np.abs(t - tsel))
        wk = w[k, :]

        ax.plot(x, np.abs(wk), color='C0', lw=2.8, ms=0, zorder=5)   # azul
        ax.plot(x, -np.abs(wk), color='C0', lw=1, ms=0)   # azul
        ax.plot(x, np.real(wk),   color='green', lw=1, ms=0, label="Re")   # rojo
        ax.plot(x, np.imag(wk),   color='C3', lw=1, ms=0, label="Im")   # verde

        ax.text(text_pos[0], text_pos[1], rf'$t={round(t[k], 1)}$', transform=ax.transAxes, fontsize=10)

        ax.set_ylim(
            min(np.min(np.real(wk)), np.min(np.imag(wk)), np.min(-np.abs(wk))) - 0.2,
            max(np.max(np.real(wk)), np.max(np.imag(wk)), np.max(np.abs(wk))) + 0.2
        )

        ax.tick_params(direction='in', top=True, right=True)

        if i < nplots - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel(r'$x$', fontsize=10)
    for ax in axes:
        ax.set_ylabel(r'$|\Psi|$', fontsize=10, rotation="horizontal", labelpad=10)

    plt.legend(fontsize=8, bbox_to_anchor = [1, 2.5])
    plt.subplots_adjust(hspace=0.08)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def gif_snapshots(x, t, w, save_path, n_frames=120, fps=20, L=None, text_pos=(0.04, 0.90)):
    frame_ids = np.unique(
        np.linspace(0, len(t) - 1, n_frames, dtype=int)
    )

    # Límites fijos para evitar que el gráfico salte durante la animación
    w_sample = w[frame_ids]
    amplitude = np.abs(w_sample)
    ymin = min(np.real(w_sample).min(), np.imag(w_sample).min(), -amplitude.max()) - 0.2
    ymax = max(np.real(w_sample).max(), np.imag(w_sample).max(), amplitude.max()) + 0.2

    fig, ax = plt.subplots(figsize=(4, 2))#figsize=(6, 4))
    #fig.subplots_adjust(left=0.18)

    envelope_up, = ax.plot([], [], color="C0", lw=2.8, ms=0)
    envelope_down, = ax.plot([], [], color="C0", lw=1, ms=0)
    real_line, = ax.plot([], [], color="green", lw=1, label="Re", ms=0)
    imag_line, = ax.plot([], [], color="C3", lw=1, label="Im", ms=0)
    time_text = ax.text(*text_pos, "", transform=ax.transAxes, fontsize=11)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\Psi|$", rotation="horizontal")#, labelpad=14)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(fontsize=9)

    if L is not None:
        ax.set_xlim(-L, L)
    else:
        ax.set_xlim(x[0], x[-1])

    def update(k):
        wk = w[k]
        amp = np.abs(wk)

        envelope_up.set_data(x, amp)
        envelope_down.set_data(x, -amp)
        real_line.set_data(x, np.real(wk))
        imag_line.set_data(x, np.imag(wk))
        time_text.set_text(rf"$t={t[k]:.1f}$")

        return envelope_up, envelope_down, real_line, imag_line, time_text

    ani = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / fps, blit=True)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return ani


def animate_solution_with_invariants(
    sim,
    save_path,
    N_frames=10,
    fps=15,
    mass=None,
    energy=None,
    theo=False,
    auto_adjust=True,
):
    x, t, w = sim.x, sim.t, sim.w

    if mass is None:
        mass = sim.get_mass()
    if energy is None:
        energy = sim.get_energy()

    delta_mass = mass - mass[0]
    delta_energy = energy - energy[0]

    frames = np.arange(0, len(t), N_frames)
    frames = np.unique(np.append(frames, len(t) - 1))

    fig, (ax_sol, ax_inv) = plt.subplots(1, 2, figsize=(6, 3))
    fig.subplots_adjust(wspace=0.1, left=0.10, right=0.97, bottom=0.16)

    # Panel izquierdo: solución
    line_sol, = ax_sol.plot(x, norm(w[0]), "-", label="Simulation", ms=0)

    if theo:
        line_theo, = ax_sol.plot(
            x, norm(sim.w_theo(x, t[0])), "--", label="Theo", alpha=0.9, 
            ms=0, color="C2"
        )
    else:
        line_theo, = ax_sol.plot([], [], "--", alpha=0.9, ms=0, color="C2")

    ax_sol.set_xlabel(r"$x$")
    ax_sol.set_ylabel(r"$|\Psi|^2$", rotation=0, labelpad=15)
    ax_sol.legend()
    title = ax_sol.set_title(rf"$t={t[0]:.2f}$")

    if not auto_adjust:
        ax_sol.set_ylim(norm(w).min() - 0.2, norm(w).max() + 0.2)

    # Panel derecho: invariantes
    line_mass, = ax_inv.plot([], [], label=r"$N(t)-N(0)$", ms=0)
    line_energy, = ax_inv.plot([], [], label=r"$E(t)-E(0)$", ms=0)

    ax_inv.set_xlim(t[0], t[-1])

    max_variation = max(
        np.max(np.abs(delta_mass)),
        np.max(np.abs(delta_energy)),
        1e-15,
    )
    ax_inv.set_ylim(-1.1 * max_variation, 1.1 * max_variation)

    ax_inv.set_xlabel(r"$t$")
    ax_inv.set_ylabel("", rotation=0, labelpad=15)
    ax_inv.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_inv.legend()

    def update(j):
        sol_j = norm(w[j])
        line_sol.set_data(x, sol_j)

        if auto_adjust:
            ax_sol.set_ylim(sol_j.min() - 0.2, sol_j.max() + 0.2)

        if theo:
            line_theo.set_data(x, norm(sim.w_theo(x, t[j])))

        line_mass.set_data(t[:j + 1], delta_mass[:j + 1])
        line_energy.set_data(t[:j + 1], delta_energy[:j + 1])
        title.set_text(rf"$t={t[j]:.2f}$")

        return line_sol, line_theo, line_mass, line_energy, title

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )

    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return ani