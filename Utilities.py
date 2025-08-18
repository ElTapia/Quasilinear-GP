import numpy as np
import os
from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import spsolve
import gzip
from matplotlib import animation


import matplotlib.pyplot as plt

# Norm function
def norm(w):
    return np.abs(w)**2

# Sech function
def sech(x):
    return 1/np.cosh(x)

# Newton method
def newton(x_0, f, f_prime, h, tol=1e-10, max_iter=1000):
    x_k = x_0
    error = np.add.reduce(np.abs(f(x_k)))
    j = 0
    while error > tol:
        #print(error)
        f_1 = -f(x_k)
        f_2 = f_prime(x_k)
        #plt.plot(norm(f_1[:len(x_k)//4]))
        #plt.plot(norm(x_k[:len(x_k)//2]))
        #plt.show()
        delta_x = spsolve(f_2, f_1, permc_spec="MMD_AT_PLUS_A")
        x_k_1 = x_k + delta_x
        error = np.sqrt(h) * np_norm(delta_x, 2) #np_norm(f(x_k_1), np.inf)
        x_k = x_k_1
        j += 1

        if j>max_iter:
            print("Simulation reaches maximum iterations. Stoping...")
            error = float('NaN')
            break

    return x_k, error, j

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

# Energy from files
def get_energy(w, h):

    N = len(w[0])
    Nt = len(w)

    energy = np.zeros(Nt)

    for n in range(Nt):
        wn = w[n]
        e = h * np.sum(1/2 * norm(np.gradient(wn, h)) + 
                       1/4 * (1 - (norm(wn)))**2 - 
                       k/4 * (np.gradient(norm(wn), h))**2)
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

    fig = plt.figure(constrained_layout=True) 
    ax = plt.axes(projection = '3d') 
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
