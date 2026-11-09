notebooks/  -  Tests, post-processing and figures
================================================

Jupyter notebooks. They are versioned WITHOUT outputs (cells must be re-run), and they write their
figures/animations (pdf, pgf, png, gif) to local folders such as Img/, which are not versioned.

How to use
    * Open each notebook from its own folder: the first cell does
      sys.path.append('../../../src/') (relative to the notebook) to import the modules.
    * Notebooks named *_files read the heavy results produced by the scripts in ../scripts/ and
      expect them under scripts/Results/<experiment>/ (see ../scripts/README.txt).
    * The other notebooks run the numerical schemes directly and are self-contained.
    * They need a working LaTeX installation (see src/config.py).

Top level
    Capacity.ipynb, Test_ODE.ipynb   Small auxiliary tests.
    GP/GP_tests.ipynb                Gross-Pitaevskii dark-soliton tests (black solitons, dt/dx).
    NLS/NLS_tests.ipynb              Tests of the cubic NLS schemes (src/NLS_Cubic).
    Comparisons/Errors_comp.ipynb    Error comparison between QGP, GP and NLS solitons.

QGP/   (quasilinear Gross-Pitaevskii equation)
    QGP_tests.ipynb                  Dark, bright and Gaussian-initial-data tests of the schemes.
    QGP_files.ipynb                  Analysis of long runs saved to files.
    Soliton_Gen_tests.ipynb          Tests of src/QGP/Soliton_Gen.py (profiles, ODE checks).
    gagliardo-nirenberg.ipynb, sum_by_parts.ipynb
                                     Numerical checks of auxiliary inequalities/identities.
    Bright_soliton/Bright_solitons.ipynb   Standing/moving bright solitons and collisions.
    Dark_soliton/Dark_solitons.ipynb       Dark solitons in D_1 and D_2.
    Antidark/Antidark_stability.ipynb      Antidark solitons.
    Cuspons_compactons/Cuspons.ipynb       Cuspons/compactons in D_3.
    Blow_up/                         Blow up of Gaussian data (needs scripts/Blow_up results):
        Gaussian_blow_up.ipynb, _A / _N / _Nt   blow-up time, amplitude A, space and time refinement.
        Blow_up_plots.ipynb          Final figures of the blow-up study.
    DS_Stability/                    Stability of dark solitons (needs scripts/DS_* results):
        DS_Stability.ipynb, A_random.ipynb, DS_stable_files.ipynb, DS_unstable_files*.ipynb,
        DS_unstability.ipynb         Stable/unstable runs, momentum, comparisons for c = c* +- n*eps.
        Eigenvals.ipynb              Eigenvalues of the linearized operator.
        Dark_solitons_DB/            DS_generator.py builds a database of stationary dark solitons
                                     over a (c, k) grid (writes DS_DB/*.npy.gz); Eigs_DS.ipynb then
                                     computes their eigenvalues. Run DS_generator.py first.
    Errors/                          Errors_files_N / _Nt: convergence in space and time and EOC
                                     tables (needs scripts/Errors results).
