src/  -  Python modules
========================

Importable code used by every script and notebook. Nothing here produces output by itself.
Scripts/notebooks add this folder to the path with sys.path.append('<../ * depth>/src/') and then
import the modules by name (e.g. "from Utilities import *", "import QGP.Defocusing.Methods as Def").

Base.py
    Abstract classes NLS_solutions (exact solutions) and NLS_numerical (grid, norms, time loop,
    saving of the solution in files, animations). All schemes inherit from them.
Utilities.py
    Norms, Newton and fixed-point solvers, error/energy computation, helpers to build the
    soliton parameters (A, B, C, D, E_D_1_3, ...), file helpers (save_file, create_folder,
    load_theo_pickle) and plotting/animation functions.
Eigenvalues.py
    Neumann finite-difference matrices and the linearized operator around a soliton, used
    for the spectral stability analysis (dark-soliton eigenvalues).
config.py
    matplotlib/LaTeX style (SciencePlots, usetex). Requires a LaTeX installation.
    It adds /Library/TeX/texbin (macOS) to PATH if dvipng is not found; edit if needed.

QGP/
    Quasilinear Gross-Pitaevskii equation.
    Focusing/Methods.py     Crank-Nicolson finite-difference scheme (focusing case).
    Defocusing/Methods.py   Crank-Nicolson finite-difference schemes (defocusing case,
                            zero and non-zero background; CNFD and Real_CNFD).
    Soliton_Gen.py          Stationary soliton profiles for given (c, k): 'dark', 'bright',
                            'anti', 'cuspon'.

NLS_Cubic/
    Cubic NLS equation (used as reference/comparison).
    Methods.py              Several schemes: CN finite differences, split-step, semi-implicit,
                            relaxation, time-splitting spectral methods.
    bright.py, dark.py      Exact bright and dark soliton solutions.
