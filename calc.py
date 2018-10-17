#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# In[ ]:


h = np.linspace(0.0, 1.0, num=1000)[1::] # exclude the first because it can diverge
kappa = np.arange(0,300)
k = np.arange(0,200)


# In[ ]:


def rho(h, alpha=1.0, x_0=0.2):
    # Weibull distribution
    return alpha/x_0*(h/x_0)**(alpha-1) * np.exp( -(h/x_0)**alpha )
    
plt.plot(h, rho(h))
dh = h[1]-h[0]
np.sum(rho(h))*dh


# In[ ]:


def p_kappa(x):
    # normal distribution (for now)
    mean = 150.0
    sigma = 1.0
    return 1.0/math.sqrt(2.0*math.pi*sigma**2) * np.exp( -(x-mean)**2/(2.0*sigma**2) )

plt.plot(kappa,p_kappa(kappa))


# In[ ]:


def r(h1, h2):
    # generalized mean
    return np.minimum(h1,h2)


# In[ ]:


def r_bar_h(h):
    # sum_{h'} rho(h') r(h,h')
    h_prime = np.copy(h)
    dh = h_prime[1] - h_prime[0]
    h_prime = h_prime.reshape( [1,h_prime.shape[0]] )
    #print(h,h_prime)
    rhh = r(h.reshape([h.shape[0],1]), h_prime)
    dr = rhh*rho(h)*dh
    #print(dr)
    return np.sum(dr,axis=1)
    
plt.plot(h, r_bar_h(h))


# In[ ]:


def r_bar():
    x = np.copy(h)
    dx = x[1]-x[0]
    y = r_bar_h(x) * rho(x)
    return np.sum(y)*dx

r_bar()


# In[ ]:


def propagator(k, h, kappa):
    # g(k|h,kappa) = \binom(kappa, k) r(h)^k ( 1-r(h))^{kappa-k}
    p = r_bar_h(h)
    _p = p.reshape([p.shape[0],1])
    #print(p, _kappa)
    b = binom(kappa, _p)
    _k = k.reshape([k.shape[0],1,1])
    #print(_k)
    return b.pmf(_k)

#k = np.array([9,10,11])
#_k = np.arange(0,100)
#h = np.array( [0.2,0.3,0.4,0.5] )
#_h = np.linspace(0.0, 1.0, num=100)[1::]
#kappa = np.array([100,110,120,130,140])
#_kappa = np.arange(0,150)
g = propagator(k, h, kappa)
print(g.shape)
#plt.plot(k, g[:,50,100])
#plt.plot(h, g[20,:,100])
plt.plot(kappa, g[30,50,:])


# In[ ]:


def k_bar_h(h):
    kappa_mean = np.sum( p_kappa(kappa) * kappa )
    #print(kappa_mean)
    return kappa_mean * r_bar_h(h)

plt.plot(h, k_bar_h(h))


# In[ ]:


def P_k(k):
    dh = h[1] - h[0]
    g = propagator(k,h,kappa)
    g_rho = g * rho(h).reshape([1,h.shape[0],1])
    gh = np.sum(g_rho, axis = 1) * dh
    gh_Pkappa = gh * p_kappa(kappa).reshape([1, kappa.shape[0]])
    return np.sum(gh_Pkappa, axis = 1)

plt.yscale("log")
plt.ylim(1.0e-4,1)
plt.plot(k, P_k(k))
P_k(k)[0:10]


# In[ ]:


n=150
p = 0.4
b = binom(n,p)
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
plt.plot(x, b.pmf(x))

