scripts/  -  Long-running simulations (background jobs)
=======================================================

Python scripts that run the expensive simulations of the paper (very fine grids and many time
steps). They were executed in the background on a computing cluster, e.g.

    nohup python -u script.py > script.out 2>&1 &

(this folder was called "Heaven_scripts" in the author's local setup).

How to run
    * Always run a script FROM ITS OWN FOLDER: it finds the modules in src/ through a relative
      sys.path.append('../../src/') (the number of "../" depends on the folder depth), and it
      creates its output folders in the current directory.
    * Outputs are .npy.gz / .pck files and logs. They are NOT versioned. To post-process them with
      the notebooks, put them under scripts/Results/<experiment>/ (the notebooks read from there).
    * Scripts named Theoretical_soliton_creation.py must be run first: they build and pickle the
      reference (theoretical) soliton that the other scripts of the same folder load.

Folders
    Blow_up/       Blow up of Gaussian initial data A*exp(-x^2/(2 sigma^2)) in the defocusing case:
                   convergence in space (blow_up_N.py) and time (blow_up_Nt.py), estimation of
                   the critical amplitude A (blow_up_A_est_sigma_*.py).
        Blow_ups/      One script per amplitude A (runs up to the blow-up time).
        Double_time/   Same amplitudes with doubled final time, and "restart_*" scripts that
                       resume a run with refined dx or dt.
        blow_up_time(deprecated).py   Old version, kept for reference.
    Bright/        Collisions of two bright solitons for several values of k (Collision_k_*.py).
    DS_Stable/     Dark-soliton stability for speeds c = c* + n*eps (stable side), with eps = 0.1:
                   one_eps/ (n=1), two_eps/ (n=2), three_eps/ (n=3). Each folder has:
                   Theoretical_soliton_creation.py (reference soliton, run first),
                   DS_stable_A_cons.py (perturbation of constant amplitude),
                   DS_stable_A_random.py (random-amplitude perturbation) and
                   control_st.py (control run without perturbation).
    DS_unstable/   Same for the unstable side, c = c* - n*eps (one_eps/, two_eps/, three_eps/), with
                   the final time tf decreasing with n. Besides the *_A_cons / *_A_random /
                   control_un scripts, ev_creator.py computes the eigenvalues/eigenvectors and
                   DS_unstable_ev.py perturbs the soliton along the unstable eigenvector.
                   Longer_time/ repeats the eigenvector experiment over a longer time interval.
    Errors/        Convergence (error vs. N and vs. Nt, EOC tables) for bright solitons
                   (Errors_bright/) and dark solitons in D_1 (Errors_dark_D1/) and D_2
                   (Errors_dark_D2/). *_L.py variants use a larger domain (L = 100 instead of 40).
    Past/          (inside DS_Stable and DS_unstable) earlier versions of the experiments.
    Antidark/      Reserved for antidark-soliton runs (no scripts yet).
