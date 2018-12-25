import sys
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
outfile = sys.argv[2]
loaded = np.loadtxt( infile, delimiter=' ')

xmin, xmax = (0.0,6.0)
plt.subplot(211)
plt.ylabel("$R_{LCC}$")
plt.xlim(xmin,xmax)
plt.plot( loaded[:,5], loaded[:,1], '-', label='ascending')
plt.plot( loaded[:,5], loaded[:,3], '-', label='descending')
plt.legend(loc='best')

plt.subplot(212)
plt.ylabel("Susceptibility")
plt.xlim(xmin,xmax)
plt.plot( loaded[:,5], loaded[:,2], '-', label='ascending')
plt.plot( loaded[:,5], loaded[:,4], '-', label='descending')
plt.xlabel("(1-f)k")

plt.savefig(outfile)
#plt.show()
