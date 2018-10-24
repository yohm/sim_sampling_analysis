#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# In[ ]:


h = np.linspace(0.0, 1.0, num=501)[1::] # exclude the first because it can diverge
dh = h[1] - h[0]
h = h - dh/2
kappa = np.arange(1,300)
k = np.arange(0,200)


# In[ ]:


def _rho(alpha=1.0, x_0=0.3):
    # Weibull distribution
    ro = alpha/x_0*(h/x_0)**(alpha-1) * np.exp( -(h/x_0)**alpha )
    return ro / (np.sum(ro)*dh)   # normalize to reduce numerical error

rho = _rho()
print(np.sum(rho)*dh)
plt.plot(h, rho)


# In[ ]:


def _p_kappa():
    # delta function
    #return np.where(kappa == 150, 1, 0)
    # binomial distribution
    _n = 5000
    _p = 0.03
    b = binom(_n, _p)
    pmf = b.pmf(kappa)
    return pmf / np.sum(pmf)   # normalize to reduce error
    # normal distribution (for now)
    #mean = 150.0
    #sigma = 30.0
    #return 1.0/math.sqrt(2.0*math.pi*sigma**2) * np.exp( -(kappa-mean)**2/(2.0*sigma**2) )

p_kappa = _p_kappa()
plt.plot(kappa,p_kappa)


# In[ ]:


def r(h1, h2):
    # generalized mean
    return np.minimum(h1,h2)
    #return np.maximum(h1,h2)
    #return np.sqrt(h1*h2)


# In[ ]:


def _r_bar_h():
    # sum_{h'} rho(h') r(h,h')
    h_prime = np.copy(h).reshape([1,h.shape[0]])
    h_ = h.reshape( [h.shape[0],1] )
    #print(h,h_prime)
    rhh = r(h_, h_prime)
    #print(rhh)
    dr = rhh*rho.reshape([1,h.shape[0]])*dh
    #print(dr)
    return np.sum(dr,axis=1)
    
r_bar_h = _r_bar_h()
print(r_bar_h.shape)
plt.plot(h, r_bar_h)


# In[ ]:


def r_bar():
    y = r_bar_h * rho * dh
    return np.sum(y)

r_bar()


# In[ ]:


def _propagator():
    # g(k|h,kappa) = \binom(kappa, k) r(h)^k ( 1-r(h))^{kappa-k}
    p = r_bar_h
    _p = p.reshape([1,p.shape[0],1])
    #print(p, _kappa)
    _kappa = kappa.reshape([1,1,kappa.shape[0]])
    b = binom(_kappa, _p)
    #print(b.shape)
    _k = k.reshape([k.shape[0],1,1])
    #print(_k)
    return b.pmf(_k)

#k = np.array([9,10,11])
#_k = np.arange(0,100)
#h = np.array( [0.2,0.3,0.4,0.5] )
#_h = np.linspace(0.0, 1.0, num=100)[1::]
#kappa = np.array([100,110,120,130,140])
#_kappa = np.arange(0,150)
g = _propagator()
print(g.shape)
plt.plot(k, g[:,300,150])
#plt.plot(h, g[20,:,100])
#plt.plot(kappa, g[30,50,:])


# In[ ]:


def _k_bar_h():
    kappa_mean = np.sum( p_kappa * kappa )
    #print(kappa_mean)
    return kappa_mean * r_bar_h

k_bar_h = _k_bar_h()
plt.plot(h, k_bar_h)


# In[ ]:


def _P_k():
    _g = g * rho.reshape([1,h.shape[0],1]) * p_kappa.reshape([1,1,kappa.shape[0]]) * dh
    return np.sum(_g, axis=(1,2))
    #gh = np.sum(g_rho, axis = 1) * dh
    #gh_Pkappa = gh * p_kappa.reshape([1, kappa.shape[0]])
    #return np.sum(gh_Pkappa, axis = 1)

P_k = _P_k()
plt.yscale("log")
plt.ylim(1.0e-4,1.0e-1)
plt.xlim(0,50)
plt.plot(k, P_k)
P_k[0:10]


# In[ ]:


# checking  <k> = \sum_{k} k P(k) = <kappa> * \bar{r}

(np.sum(k * P_k), r_bar() * np.sum(kappa*p_kappa))


# In[ ]:


# <k^2> = \sum_{k} k^2 P(k)
#   = <kappa>rbar + <kappa(kappa-1)> \sum_{h} rbar(h)^2 rho(h)

_x = np.sum(k * k * P_k)
_y = np.sum(kappa * p_kappa) * r_bar() + np.sum(kappa*(kappa-1)*p_kappa) * np.sum(r_bar_h*r_bar_h*rho)*dh
(_x, _y)


# In[ ]:


# degree correlation

def _kappa_nn_bar_kappa():
    # \bar{\kappa}_{nn}(\kappa) = \sum_{\kappa'} \kappa' p_o(\kappa' | \kappa)
    # tentatively assume P(kappa'|kappa) = P(kappa)
    kappa_mean = np.sum(kappa * p_kappa)
    return np.full(kappa.shape, kappa_mean)

kappa_nn_bar_kappa = _kappa_nn_bar_kappa()
plt.plot(kappa, kappa_nn_bar_kappa)


# In[ ]:


def _r_nn_h():
    # h, h_prime are axis=0,1, respectively.
    nh = h.shape[0]
    h_prime = np.copy(h).reshape([1,nh])
    rho_h_prime = rho.reshape([1,nh])
    r_bar_h_prime = r_bar_h.reshape([1,nh])
    r_bar_h_ = r_bar_h.reshape( [nh,1] )
    h_ = h.reshape( [nh,1] )
    x = r( h_, h_prime ) * rho_h_prime * r_bar_h_prime / r_bar_h_
    return np.sum( x, axis=1 ) * dh
    
r_nn_h = _r_nn_h()
plt.plot(h, r_nn_h)
plt.plot(h, r_bar_h)


# In[ ]:


def _k_nn_bar_k():
    # k, h, kappa are axis=0,1,2, respectively
    nk = k.shape[0]
    nh = h.shape[0]
    nkappa = kappa.shape[0]
    P_k_ = P_k[ P_k > 0 ]
    p_k_ = P_k_.reshape( [P_k_.shape[0],1,1,] )
    _g = g[P_k > 0,:,:]
    #print(p_k_.shape)
    rho_h_ = rho.reshape( [1,nh,1] )
    p_kappa_ = p_kappa.reshape( [1,1,nkappa] )
    r_nn_h_ = r_nn_h.reshape( [1,nh,1] )
    kappa_nn_bar_kappa_ = kappa_nn_bar_kappa.reshape( [1,1,nkappa] )
    #g_p_k_ = np.where( p_k_ > 0.0, g/p_k_, 0.0)
    return 1 + np.sum( _g / p_k_ * rho_h_ * p_kappa_ * r_nn_h_ * (kappa_nn_bar_kappa_), axis=(1,2) ) * dh
    #return 1 + np.sum( _g / p_k_ * rho_h_ * p_kappa_ * r_nn_h_ * (kappa_nn_bar_kappa_-1), axis=(1,2) ) * dh

k_nn_bar_k = _k_nn_bar_k()
#print(k_nn_bar_k)
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.plot(k[P_k > 0], k_nn_bar_k)


# In[ ]:


np.sum(P_k[P_k > 0] * k_nn_bar_k)
np.sum(k * P_k)
#P_k[ P_k > 0 ].shape


# In[ ]:


# clustering coefficient

def _p_hprime_given_h():
    # h,h' are axis-0,1
    # p(h'|h) = r(h',h) rho(h') / r_bar(h)
    nh = h.shape[0]
    h_ = h.reshape( (nh,1) )
    h_prime = h.reshape( (1,nh) )
    rho_hprime = rho.reshape( (1,nh) )
    rbar_h = r_bar_h.reshape( (nh,1) )
    return r(h_, h_prime) * rho_hprime / rbar_h

p_hprime_given_h = _p_hprime_given_h()
#plt.plot(h, p_hprime_given_h[:,100])
#h_bar_given_h = np.sum( p_hprime_given_h * h.reshape( (1,h.shape[0]) ), axis=1) * dh
#plt.plot(h, h_bar_given_h)

def _c_h():
    # h, h', h'' are axis-0,1,2, respectively
    # \sum_{h', h''} = r(h', h'') * p(h'|h) * p(h''|h)
    nh = h.shape[0]
    h_ = h.reshape( (nh,1,1) )
    h_prime = h.reshape( (1,nh,1) )
    h_prime2 = h.reshape( (1,1,nh) )
    p_hprime_given_h_ = p_hprime_given_h.reshape( (nh,nh,1) )
    p_hprime2_given_h_ = p_hprime_given_h.reshape( (nh,1,nh) )
    r_ = r(h_prime, h_prime2).reshape( (1,nh,nh) )
    return np.sum( r_ * p_hprime_given_h_ * p_hprime2_given_h_, axis=(1,2) ) * dh * dh

c_h = _c_h()
#plt.plot(h, c_h)

def _c_k():
    # k, h, kappa are axis-0,1,2, respectively
    # 1/P(k) * \sum_{h,\kappa} g(k|h,\kappa) rho(h) P(\kappa) c_h c_o(\kappa)
    nh = h.shape[0]
    nk = k.shape[0]
    nkappa = kappa.shape[0]
    _rho_h = rho.reshape( (1,nh,1) )
    _p_kappa = p_kappa.reshape( (1,1,nkappa) )
    _c_h = c_h.reshape( (1,nh,1) )
    #_c_o_kappa = 0.1
    # tentatively assume c_o(kappa) = 0.1 * kappa^{-1}
    # _c_o_kappa = 0.1 / (kappa**2)
    _c_o_kappa = 0.03
    return 1.0 / P_k * np.sum( g * _rho_h * _p_kappa * _c_h * _c_o_kappa, axis=(1,2) ) * dh

c_k = _c_k()
plt.yscale("log")
plt.xscale("log")
plt.plot(k, c_k)

#c_h_kappa = _c_h_kappa()
#plt.plot(h, c_h_kappa[:,150])


# In[ ]:


# Pearson's correlation coefficient, but something is wrong...

def _pearson():
    _kappa_mean = np.sum( kappa * p_kappa )
    _r_bar = r_bar()
    _kappa_kappa_prime = 150*149    # [TODO] tentative implementation: degree correlation in the original network
    _kappa_kappa_mean = np.sum( kappa * (kappa-1) * p_kappa )
    _r2_bar = np.sum( r_bar_h * r_bar_h * rho ) * dh
    print(_r2_bar, _r_bar*_r_bar)
    _numerator = _kappa_mean * _r_bar + _kappa_kappa_prime * _r2_bar - _kappa_mean * _kappa_mean * _r_bar * _r_bar
    #_numerator = _kappa_kappa_prime * _r2_bar - _kappa_mean * _kappa_mean * _r_bar * _r_bar
    _denominator = _kappa_mean * _r_bar + _kappa_kappa_mean * _r2_bar - _kappa_mean * _kappa_mean * _r_bar * _r_bar
    print(_kappa_mean * _r_bar + _kappa_kappa_mean * _r2_bar, (_kappa_mean * _r_bar)**2 )
    return (_numerator, _denominator)


_pearson()


# In[ ]:





# In[ ]:


# comparison with simulation
def _compare_pk():
    path = "/Users/murase/work/oacis/public/Result_development/5bcd5bb6d12ac6187c093163/5bd00a02d12ac61729031987/5bd00bd1d12ac617290319fe/degree_distribution_ave.dat"
    d = np.loadtxt(path)
    plt.yscale("log")
    plt.ylim(1.0e-4,4.0e-2)
    plt.xlim(0,70)
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.plot(k, P_k )
    plt.plot(d[:,0], d[:,1]/5000)
    plt.savefig("pk_sim.pdf")
    
_compare_pk()
    
    


# In[ ]:


def _compare_knn():
    path = "/Users/murase/work/oacis/public/Result_development/5bcd5bb6d12ac6187c093163/5bd00a02d12ac61729031987/5bd00bd1d12ac617290319fe/neighbor_degree_correlation_ave.dat"
    d = np.loadtxt(path)
    plt.xscale("log")
    plt.xlim(1.0e0, 1.0e2)
    plt.xlabel("k")
    plt.ylabel("k_nn(k)")
    plt.plot(k[P_k > 0], k_nn_bar_k)
    plt.plot(d[:,0], d[:,1])
    plt.savefig("knn_sim.pdf")
    
_compare_knn()


# In[ ]:


def _compare_ck():
    path = "/Users/murase/work/oacis/public/Result_development/5bcd5bb6d12ac6187c093163/5bd00a02d12ac61729031987/5bd00bd1d12ac617290319fe/cc_degree_correlation_ave.dat"
    d = np.loadtxt(path)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("c(k)")
    plt.plot(k[2:], c_k[2:])
    plt.plot(d[:,0], d[:,1])
    plt.savefig("ck_sim.pdf")
    
_compare_ck()


# In[ ]:




