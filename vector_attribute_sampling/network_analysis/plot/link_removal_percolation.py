import sys
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
outfile = sys.argv[2]
loaded = np.loadtxt( infile, delimiter=' ')

plt.subplot(211)
plt.ylabel("$R_{LCC}$")
plt.plot( loaded[:,0], loaded[:,1], '-', label='ascending')
plt.plot( loaded[:,0], loaded[:,3], '-', label='descending')
plt.legend(loc='best')

plt.subplot(212)
plt.ylabel("Susceptibility")
plt.plot( loaded[:,0], loaded[:,2], '-', label='ascending')
plt.plot( loaded[:,0], loaded[:,4], '-', label='descending')
plt.xlabel("f")

plt.savefig(outfile)
#plt.show()

