#import os 
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import os, shutil

tex_bin = os.path.expanduser("~/Library/TinyTeX/bin/universal-darwin")
if shutil.which("dvipng") is None:            # sólo si todavía no lo ve
    os.environ["PATH"] = tex_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib.pyplot as plt


import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

import scienceplots
plt.style.use(['science', 'grid']) #'ieee'

plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'mathtext.fontset': 'cm', #'stix',
    'font.family': 'serif',
    "font.serif": ["Computer Modern Roman"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.formatter.use_mathtext':True
})

plt.rcParams['figure.dpi'] = 150
plt.rcParams["font.serif"] = "Times"
plt.rcParams["font.size"] = 8

np.set_printoptions(legacy="1.25")