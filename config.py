#import os 
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import os, shutil

tex_bin = os.path.expanduser("/Library/TeX/texbin") #"~/Library/TinyTeX/bin/universal-darwin")
if shutil.which("dvipng") is None:            # sólo si todavía no lo ve
    os.environ["PATH"] = tex_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib.pyplot as plt


import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

import scienceplots
plt.style.use(['science', 'grid']) #'ieee'

#plt.rcParams.update({
#    'mathtext.fontset': 'cm', #'stix',
#    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
#    'font.family': 'serif',
#    "font.serif": ["Computer Modern Roman"],
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#    'axes.formatter.use_mathtext':True,
#    "grid.linewidth" : .5,
#    "grid.alpha": .1
#})

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,upgreek}",
    "pgf.preamble": r"\usepackage{amsmath,amssymb,upgreek}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "pgf.rcfonts": False,
    "axes.formatter.use_mathtext": True,
    "grid.linewidth": .5,
    "grid.alpha": .1,
})


plt.rcParams['figure.dpi'] = 150
#plt.rcParams["font.serif"] = "Times"
plt.rcParams["font.size"] = 8

np.set_printoptions(legacy="1.25")
