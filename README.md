# Quasilinear-GP

Numerical schemes and numerical experiments for the **quasilinear Gross–Pitaevskii (QGP) equation**
(focusing and defocusing cases) and, for comparison, for the cubic nonlinear Schrödinger (NLS) and
Gross–Pitaevskii equations.

The repository contains code only: simulation outputs, figures and animations are not included,
and notebooks are stored without outputs.

## Contents

```
Quasilinear-GP/
├── src/          Python modules: numerical schemes, soliton generators, utilities, eigenvalue tools
│   ├── QGP/          Crank–Nicolson finite-difference schemes (Focusing/, Defocusing/) and soliton profiles
│   └── NLS_Cubic/    Schemes and exact solutions for the cubic NLS equation
├── scripts/      Long-running simulations (blow-up, soliton collisions, dark-soliton stability, errors)
└── notebooks/    Tests, post-processing and figures
```

Each folder has a `README.txt` describing its contents.

| Module | Content |
| --- | --- |
| `src/Base.py` | Base classes for exact and numerical solutions (grids, norms, saving, animation). |
| `src/QGP/Focusing/Methods.py`, `src/QGP/Defocusing/Methods.py` | Crank–Nicolson finite-difference schemes for the focusing / defocusing QGP equation. |
| `src/QGP/Soliton_Gen.py` | Stationary bright, dark, anti-dark soliton and cuspon profiles. |
| `src/NLS_Cubic/` | Finite-difference, split-step, relaxation and time-splitting spectral schemes for the cubic NLS. |
| `src/Eigenvalues.py` | Linearization around a soliton and eigenvalue computation (stability analysis). |
| `src/Utilities.py` | Norms, nonlinear solvers, error and energy computation, plotting helpers. |

## Requirements

Python ≥ 3.10 with the packages in `requirements.txt` (NumPy, SciPy, Matplotlib, ...). Plots use
LaTeX rendering, so a TeX distribution is needed to generate the figures.

Scripts and notebooks import the modules through a relative path to `src/`, so they should be run
from their own folder.

## License

MIT — see [LICENSE](LICENSE).
