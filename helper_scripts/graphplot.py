import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


x = np.linspace(0, 1, 100)
asym1 = -0.8 / ((x+np.sqrt(0.1))**2) + 8
asym2 = -1 / ((x+np.sqrt(0.5))**4) + 10
pl.xlim(0, 1)
pl.ylim(0, 11)
pl.plot(x, asym1, color="black",  linewidth=2, linestyle="-")
pl.plot(x, asym2, color="red",  linewidth=2, linestyle="-")
pl.show()


