# Generates dark soliton or anti-dark soliton for QGP
import numpy as np
import warnings
from scipy.optimize import minimize_scalar
from scipy.interpolate import make_interp_spline


def soliton(c,
            k,
            soliton_type='dark',
            L=10,
            N=5_000,
            x=None,
            as_func=False):

    r"""Obtain a soliton for the QGP equation.

    Returns the numerical soliton with a giving c and k. It may return a
    callable or the soliton in certain space points, depending on 'x' argument.

    In order for the soliton to exist, the pair should leave in the
    corresponding set (D_1, D_2 or D_3) and also be consistent with the
    existence of certain types of solitons in each set. For example, dark
    solitons leaves only in D_1 and D_2 but not in D_3.

    This function calculates an inverse of a function depending on whether the
    type of soliton is required. By default, the domain of the function is
    obtained depending on c and k, and is compared with the type of the soliton.

    Parameters
    ----------
    c : float
        Value representing the speed of the soliton. It should be positive and
        consistent with the type of the required soliton. When soliton is bright,
        c is equal to w in the theoretical formula.

    k : float
        Real value representing a parameter of the QGP equation. It should be
        consistent with the type of the required soliton and also with the
        soliton velocity.

    soliton_type: {'dark', 'anti', 'cuspon'}, default: 'dark'
        Type of the soliton.
        - 'dark': Generates a Dark soliton. Lives in D_1 and D_2.
        - 'anti': Generates an Anti Dark soliton. Lives in D_3.
        - 'cuspon': TODO: COMPLETE

    L: float, default: 10
        Value representig the width of the spatial domain. It should be
        positive as the domain is symmetric respect to 0.

    N: int, default: 5e4
        Value representing the total of discretization point.
        
    x : array_like, default: np.linspace(-L, L, N)
        Array representing the space domain to generates the soliton. The array
        should be symmetric respecto to 0.

    as_func: bool, default: False
        Whether the return is an array or a callable.

    Returns
    -------
    callable or ndarray
        Soliton profile with parameters c and k. It can take only arrays.

    """

    print("Creating soliton...")

    i = 1j

    D_bright = [[0, np.inf], [0, np.inf]]
    D_1 = [[0, np.sqrt(2)], [0, 1/2]]
    D_2 = [[0, np.sqrt(2)], [-np.inf, 0]]
    D_3 = [[np.sqrt(2), np.inf], [1/2, np.inf]]


    if soliton_type!="dark" and soliton_type!="anti" and soliton_type!="cuspon" and soliton_type!="bright":
        raise ValueError("No soliton exists for %s" %soliton_type)

    is_in_D1 = is_in_D(c, k, D_1, open_domain=[[False, True], [True, True]])
    is_in_D2 = is_in_D(c, k, D_2, open_domain=[[False, True], [True, False]])
    is_in_D3 = is_in_D(c, k, D_3, open_domain=[[True, True], [True, True]])
    is_in_D_bright = is_in_D(c, k, D_bright, open_domain=[[True, True], [False, True]])

    dark_cond = is_in_D1 or is_in_D2
    anti_cond = is_in_D3
    bright_cond = is_in_D_bright
    
    not_dark_cond = soliton_type == "dark" and not dark_cond
    not_anti_cond = (soliton_type == "anti" and not anti_cond)
    not_bright_cond = (soliton_type == "bright" and not bright_cond)

    cond = not_dark_cond or not_anti_cond or not_bright_cond
    
    if cond:
        raise ValueError("Pair (%s, %g) is not in the soliton domain" %(round(c, 3), round(k, 3)))

    # Domain
    ymin = 0
    ymax = 1-c*c/2
    
    if soliton_type=="anti":
        ymin = 1-c*c/2
        ymax = 0

    if soliton_type=="bright":
        ymin = 0
        ymax = np.sqrt(2*c)


    ref1, ref2 = _get_valid_refpoints(ymin, ymax)
    
    if x is None:
        x = np.linspace(-L, L, N)

    m = N//2
    xpos = x[m:]


    def Func_c_k(y):

        y = np.asarray(y)

        valid_mask = (y > ymin) & (y <= ymax)

        if soliton_type=="anti":
            valid_mask = (y >= ymin) & (y < ymax)
            
        if soliton_type=="bright":
            valid_mask = (y > ymin) & (y < ymax)

        def f_c_k(y):
            root = np.sqrt((2-c*c-2*y)/(1-2*k+2*k*y))
            sum_1 = 2 * np.sqrt(k) * np.arctan(np.sqrt(k) * root)
            sum_2 = 2 * np.sqrt((1-2*k)/(2-c*c)) * np.arctanh(np.sqrt((1-2*k)/(2-c*c)) * root)
            output = sum_1 + sum_2

            return output


        def f_w_k(y):
            root = np.sqrt((2 * c - y*y)/(1+2*k*y*y))
            sum_1 = 1 / np.sqrt(c) * np.arctanh(1/np.sqrt(2*c) * root)
            sum_2 = 2 * np.sqrt(k) * np.arctan(np.sqrt(2*k) * root)
            output = sum_1 + sum_2

            return output
    

        def g_c_k(y):
            root = np.sqrt((2-c*c-2*y)/(1-2*k+2*k*y))
            sum_1 = -2 * np.sqrt(-k) * np.arctanh(np.sqrt(-k) * root)
            sum_2 = 2 * np.sqrt((1-2*k)/(2-c*c)) * np.arctanh(np.sqrt((1-2*k)/(2-c*c)) * root)
            output = sum_1 + sum_2
            return output


        if (soliton_type == "dark" and is_in_D1) or (soliton_type == "anti" and is_in_D3):
            func_c_k = f_c_k

        elif soliton_type == "dark" and is_in_D2:
            func_c_k = g_c_k
        
        elif soliton_type == "bright" and is_in_D_bright:
            func_c_k = f_w_k

        # Increasing or decreasing trend
        trend = np.sign(func_c_k(ref2) - func_c_k(ref1))

        # Assign infinite to values outside the domain
        val = np.full_like(y, -np.inf*trend, dtype=np.float64)
        val[y > ymax] = np.inf * trend

        val[valid_mask] = func_c_k(y[valid_mask])

        return val


    def eta_c_k():

        min_kwargs = {}
        min_kwargs['bracket'] = (ref1, ref2)
        min_kwargs['tol'] = 1.48e-12
        min_kwargs['method'] = 'Golden'

        results = np.full_like(x, np.nan, dtype=np.float64)
        resultsmask = np.zeros(x. shape, dtype=bool)

        for j in range(xpos.size):

            if Func_c_k(ymax) == xpos[j]:
                results[m+j] = ymax

            if Func_c_k(ymin) == xpos[j]:
                results[m+j] = ymin

            optimizer = (lambda x, j=j,
                            bounded_f=Func_c_k: (np.abs(Func_c_k(x) - xpos[j]))**2)

            try:
                with warnings.catch_warnings(record=True):
                    result = minimize_scalar(optimizer, **min_kwargs)

                results[m+j] = result.x
                resultsmask[m+j] = result.success

            except:
                resultsmask[m+j] = False

        if any(~resultsmask[m:]):
                warnings.warn("Trouble calculating inverse for values: "
                              "%s" % str(xpos[~resultsmask[m:]]), RuntimeWarning)   
        try:
            np.testing.assert_array_almost_equal(xpos, Func_c_k(results[m:]), decimal=2)

        except AssertionError:
            warnings.warn("Results obtained with less than %g "
                              "decimal digits of accuracy"
                              % 2, RuntimeWarning)

        results[:m] = np.flip(results[m:])
        return results


    def rho_c_k(eta):
        output = np.sqrt(1 - eta)
        return output


    def theta_c_k(eta):

        ## Function to integrate
        integrand = eta/(1 - eta)

        theta = np.zeros(N)

        dx = xpos[1] - xpos[0]

        theta[m] = 0

        for j in range(m-1):

            theta[j+1+m] = theta[j+m] + c/2 * (dx/2 * (integrand[j+m] + integrand[j+m+1]))

        theta[:m] = -np.flip(theta[m:])

        return theta


    def u_c_k():
        i = 1j

        eta = eta_c_k()

        if c == 0:
            output = np.sign(x) * np.sqrt(1 - eta)
            return output + 0*i
            

        rho = rho_c_k(eta)
        theta = theta_c_k(eta)
        return rho * np.exp(i*theta)


    if soliton_type == "bright" and is_in_D_bright:
        def u_c_k():
            i = 1j
            output = eta_c_k()
            return output + 0*i
        
    if k==0 and soliton_type=="dark" and as_func:
        
        def dark_soliton(x):
            A = np.sqrt((2-c**2)/2)
            sum_1 = A * np.tanh(np.sqrt(2-c**2)/2 * x)
            sum_2 = i * c/np.sqrt(2)

            return sum_1 - sum_2
    
        print("Soliton created")
        return dark_soliton, x
    
    soliton_arr = u_c_k()

    print("Soliton created")

    if as_func:
        soliton_func = make_interp_spline(x, soliton_arr)
        return soliton_func, x

    return soliton_arr, x


def is_in_D(c, k, D, open_domain):
    c_open, k_open = open_domain
    c_domain, k_domain = D

    c_cond = is_in_interval(c, c_domain, c_open)
    k_cond = is_in_interval(k, k_domain, k_open)

    return c_cond and k_cond


def is_in_interval(x, D, open_domain):

    xmin, xmax = D
    xmin_open, xmax_open = open_domain

    if xmax_open:
        max_cond = x < xmax

    else:
        max_cond = x <= xmax

    if xmin_open:
        min_cond = xmin < x

    else:
        min_cond = xmin <= x

    return min_cond and max_cond


def _get_valid_refpoints(xmin, xmax):
    d = xmax - xmin
    ref1 = xmin + d / 4.
    ref2 = xmax - d / 4.
    return ref1, ref2

