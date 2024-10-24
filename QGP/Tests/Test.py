import matplotlib.pyplot as plt
import numpy as np
from Soliton_Gen import soliton, norm

c = 2 #np.sqrt(2)/2
k = 3 #0.0625
sol = soliton(c, k, soliton_type="anti", L=20)
plt.plot(norm(sol))
plt.show()