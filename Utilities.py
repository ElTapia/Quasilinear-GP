import numpy as np
from scipy.sparse.linalg import spsolve

# Norm function
def norm(w):
    return np.abs(w)**2


# Newton method
def newton(x_0, f, f_prime, tol=1e-8):
    x_k = x_0
    error = np.add.reduce(np.abs(f(x_k)))
    while error > tol:
        f_1 = -f(x_k)
        f_2 = f_prime(x_k)
        delta_x = spsolve(f_2, f_1, permc_spec="MMD_AT_PLUS_A")
        x_k = x_k + delta_x
        error = np.add.reduce(np.abs(f(x_k)))

    return x_k, error