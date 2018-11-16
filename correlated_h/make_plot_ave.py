import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = np.loadtxt("pk_ave.dat")
plt.errorbar(a[:,0], a[:,1], yerr=a[:,2])
plt.savefig("pk_ave.png")
plt.clf()

a = np.loadtxt("knn_ave.dat")
plt.errorbar(a[:,0], a[:,1], yerr=a[:,2])
plt.savefig("knn_ave.png")
plt.clf()

a = np.loadtxt("ck_ave.dat")
plt.errorbar(a[:,0], a[:,1], yerr=a[:,2])
plt.savefig("ck_ave.png")
plt.clf()
