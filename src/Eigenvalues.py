import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Utilities import *

def D1_neumann_centered(N, h):
    """
    First-derivative matrix with centered differences
    and homogeneous Neumann boundary conditions.

    At the interior nodes:
        u_x(x_j) ≈ (u_{j+1} - u_{j-1}) / (2h)

    At the boundaries, using reflection for homogeneous Neumann conditions:
        u_{-1} = u_1,   u_{N} = u_{N-2}
    which gives
        u_x(x_0) = 0,
        u_x(x_{N-1}) = 0.

    Parameters
    ----------
    N : int
        Number of grid points.
    h : float
        Spatial step size.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse NxN matrix.
    """
    e = np.ones(N)

    D1 = sp.diags(
        diagonals=[-e, e],
        offsets=[-1, 1],
        shape=(N, N),
        format="lil"
    )

    D1 /= 2*h

    # Boundaries: zero derivative from homogeneous Neumann conditions with reflected extension
    D1[0, :] = 0.0
    D1[-1, :] = 0.0

    return D1.tocsr()


def D2_neumann_centered(N, h):
    """
    Second-derivative matrix with homogeneous Neumann conditions,
    using centered differences at the boundaries as well.

    The resulting matrix is:

        1/h^2 * [ -2   2   0  ...  0
                   1  -2   1  ...  .
                   0   1  -2  ...  0
                   .   .   .  ...  1
                   0  ...  0   2  -2 ]

    Parameters
    ----------
    N : int
        Number of grid points.
    h : float
        Spatial step size.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse NxN matrix.
    """
    e = np.ones(N)

    D2 = sp.diags(
        diagonals=[e, -2.0 * e, e],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="lil"
    )

    # Boundary adjustments for homogeneous Neumann conditions with a centered scheme
    D2[0, 1] = 2.0
    D2[-1, -2] = 2.0

    return D2.tocsr() / h**2


def build_linearized_matrix(us, D1, D2, k, c, profs=None):

    u = np.asarray(us)
    uc = np.conjugate(u)

    N = us.size

    if profs:
        eta, theta = profs

        eta_x = - eta * np.sqrt((-2 * eta + (2 - c**2))/(1 - 2 * k + 2 * k * eta))
        eta_xx = (- 3 * eta**2 + (2 - c**2) * eta - k * eta_x**2)/(1 - 2 * k + 2 * k * eta)

        ux = np.exp(1j * theta) / (2 * np.sqrt(1-eta)) * (-eta_x + 1j * c * eta)

        uxx = np.exp(1j * theta) / (2 * np.sqrt(1-eta)) * (eta / (2*(1-eta)) * (-eta_x + 1j * c * eta) + 1j * c * eta / (2 * (1-eta)) * (-eta_x + 1j * c * eta) - eta_xx + 1j * c * eta_x)

        unxx = -eta_xx

    else:  
        ux = D1 @ u
        uxx = D2 @ u

        ucx = np.conjugate(ux) #D1 @ uc
        ucxx = np.conjugate(uxx) #D2 @ uc

        unxx = uxx * uc + 2 * ux * ucx + u * ucxx #D2 @ norm(u)

    d1_A = 1j * c - 2.0 * k * u * ucx

    # D2
    d2_A = 1 - k * norm(u)

    # Multiplicative part of A
    a0 = 1.0 - 2.0 * norm(u) - k * u * ucxx - k * unxx

    d1_B = 2 * k * u * ux

    # D2
    d2_B = k * u**2

    # Multiplicative part of B
    b0 = -u**2 - k * u * uxx

    I = sp.identity(N, format="csr")

    A = (
          sp.diags(a0, 0, shape=(N, N), format="csr")
        + sp.diags(d1_A, 0, shape=(N, N), format="csr") @ D1
        + sp.diags(d2_A, 0, shape=(N, N), format="csr") @ D2
    )

    B = (
         sp.diags(b0, 0, shape=(N, N), format="csr")
        - sp.diags(d1_B, 0, shape=(N, N), format="csr") @ D1
        - sp.diags(d2_B, 0, shape=(N, N), format="csr") @ D2
    )


    M = sp.bmat(
        [[np.imag(A+B),          np.real(A-B)],
         [ -np.real(A+B),        np.imag(A-B)]],
        format="csr"
    )
    
    #M = sp.bmat(
    #    [[None,          A-B],
    #     [ -(A+B),  None]],
    #    format="csr"
    #)

    #M = sp.bmat(
    #    [[-1j * A,          -1j * B],
    #     [ 1j * B.conjugate(), 1j * A.conjugate()]],
    #    format="csr"
    #)

    return M, A, B


def get_eta(evals, evecs, N, imag_tol=1e-8):

    # Indices of eigenvalues with imaginary part close to zero
    real_like_idx = np.where(np.abs(np.imag(evals)) < imag_tol)[0]
    
    if real_like_idx.size == 0:
        raise RuntimeError("No eigenvalues found with imaginary part close to zero.")
    
    # Among the nearly real eigenvalues, pick the one with largest real part
    idx = real_like_idx[np.argmax(np.real(evals[real_like_idx]))]
    
    lam = evals[idx]
    vec = evecs[:, idx]
    
    p = vec[:N]
    q = vec[N:]
    
    print("Largest non-complex eigenvalue =", lam)
    
    eta = p + 1j*q

    return eta
