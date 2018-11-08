#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# In[ ]:


h = np.linspace(0.0, 1.0, num=501)[1::] # exclude the first because it can diverge
h = h - (h[1]-h[0])/2
kappa = np.arange(1,300)

def _rho(h, alpha=1.0, x_0=0.3):
    # Weibull distribution
    ro = alpha/x_0*(h/x_0)**(alpha-1) * np.exp( -(h/x_0)**alpha )
    dh = h[1]-h[0]
    return ro / (np.sum(ro)*dh)   # normalize to reduce numerical error

rho = _rho(h)
plt.plot(h, rho)


# In[ ]:


def _p_kappa(kappa):
    # delta function
    #return np.where(kappa == 150, 1, 0)
    # binomial distribution
    _n = 5000
    _p = 0.03
    b = binom(_n, _p)
    pmf = b.pmf(kappa)
    return pmf / np.sum(pmf)   # normalize to reduce error

P_kappa = _p_kappa(kappa)
plt.plot(kappa,P_kappa)


# In[ ]:


def gen_mean(x,y,beta):  # when beta<=-10 or >=10, minimum or maximum is used
    if beta == 0:
        return np.sqrt(x*y)
    elif beta <= -10:
        return np.minimum(x,y)
    elif beta >= 10:
        return np.maximum(x,y)
    else:
        return ((x**beta+y**beta)/2)**(1.0/beta)

r = lambda x, y: gen_mean(x,y,-10)

r(1.0,3.0)


# In[ ]:


class NodalSampling:
    
    def __init__(self, h, kappa, rho_h, P_kappa, r):
        self.h = h
        self.nh = h.shape[0]
        self.dh = self.h[1]-self.h[0]
        self.kappa = kappa
        self.nkappa = kappa.shape[0]
        self.rho_h = rho_h
        assert( rho_h.shape == h.shape )
        self.P_kappa = P_kappa
        assert( P_kappa.shape == kappa.shape )
        self.r = r
        self.k = np.arange(0, np.amax(self.kappa)+1)
        self.nk = self.k.shape[0]
        self.results = {
            "r_bar_h": None,
            "r_bar": None,
            "g": None,
            "P_k": None,
            "r_nn_h": None,
            "p_hprime_given_h": None,
            "c_h": None
        }
        
    def r_bar_h(self):
        # sum_{h'} rho(h') r(h,h')
        if self.results["r_bar_h"] is not None:
            return self.results["r_bar_h"]
        h_prime = np.copy(self.h).reshape([1,self.nh])
        h_ = self.h.reshape( [self.nh,1] )
        rhh = self.r(h_, h_prime)
        dr = rhh*rho.reshape([1,self.nh])*self.dh
        self.results["r_bar_h"] = np.sum(dr,axis=1)
        return self.results["r_bar_h"]
    
    def r_bar(self):
        # sum_{h,h'} rho(h') rho(h) r(h,h')
        if self.results["r_bar"] is not None:
            return self.results["r_bar"]
        y = self.r_bar_h() * self.rho_h * self.dh
        self.results["r_bar"] = np.sum(y)
        return self.results["r_bar"]
    
    def g(self):
        # g(k|h,kappa) = \binom(kappa, k) r(h)^k ( 1-r(h))^{kappa-k}
        #   k,h,kappa are 0th,1st,2nd axis, respectively
        if self.results["g"] is not None:
            return self.results["g"]
        _p = self.r_bar_h().reshape([1,self.nh,1])
        _kappa = self.kappa.reshape([1,1,self.nkappa])
        b = binom(_kappa, _p)
        _k = self.k.reshape([self.nk,1,1])
        self.results["g"] = b.pmf(_k)
        return self.results["g"]
    
    def P_k(self):
        if self.results["P_k"] is not None:
            return self.results["P_k"]
        _g = g * self.rho_h.reshape([1,self.nh,1]) * self.P_kappa.reshape([1,1,self.nkappa])
        self.results["P_k"] = np.sum(_g, axis=(1,2)) * self.dh
        return self.results["P_k"]
    
    def r_nn_h(self):
        # h, h_prime are axis=0,1, respectively.
        if self.results["r_nn_h"] is not None:
            return self.results["r_nn_h"]
        h_prime = np.copy(self.h).reshape([1,self.nh])
        rho_h_prime = self.rho_h.reshape([1,self.nh])
        r_bar_h_prime = self.r_bar_h().reshape([1,self.nh])
        r_bar_h = self.r_bar_h().reshape( [self.nh,1] )
        h = self.h.reshape( [self.nh,1] )
        x = r( h, h_prime ) * rho_h_prime * r_bar_h_prime / r_bar_h
        self.results["r_nn_h"] = np.sum( x, axis=1 ) * self.dh
        return self.results["r_nn_h"]
    
    def k_nn_k(self, kappa_nn):
        # k, h, kappa are axis=0,1,2, respectively
        assert( kappa_nn.shape == self.kappa.shape )
        Pk = self.P_k()
        Pk_ = Pk[ Pk > 0 ]
        Pk_ = Pk_.reshape( [Pk_.shape[0],1,1,] )
        _g = g[Pk > 0,:,:]
        rho_h_ = self.rho_h.reshape( [1,self.nh,1] )
        p_kappa_ = self.P_kappa.reshape( [1,1,self.nkappa] )
        r_nn_h_ = self.r_nn_h().reshape( [1,self.nh,1] )
        kappa_nn_ = kappa_nn.reshape( [1,1,self.nkappa] )
        return 1 + np.sum( _g / Pk_ * rho_h_ * p_kappa_ * r_nn_h_ * (kappa_nn_-1), axis=(1,2) ) * self.dh;
    
    def _p_hprime_given_h(self):
        # h,h' are axis-0,1
        # p(h'|h) = r(h',h) rho(h') / r_bar(h)
        if self.results["p_hprime_given_h"] is not None:
            return self.results["p_hprime_given_h"]
        h_ = self.h.reshape( (self.nh,1) )
        h_prime = self.h.reshape( (1,self.nh) )
        rho_hprime = self.rho_h.reshape( (1,self.nh) )
        rbar_h = self.r_bar_h().reshape( (self.nh,1) )
        self.results["p_hprime_given_h"] = r(h_, h_prime) * rho_hprime / rbar_h
        return self.results["p_hprime_given_h"]
    
    def c_h(self):
        # h, h', h'' are axis-0,1,2, respectively
        # \sum_{h', h''} = r(h', h'') * p(h'|h) * p(h''|h)
        if self.results["c_h"] is not None:
            return self.results["c_h"]
        nh = self.nh
        h_ = self.h.reshape( (nh,1,1) )
        h_prime = self.h.reshape( (1,nh,1) )
        h_prime2 = self.h.reshape( (1,1,nh) )
        p_hprime_given_h_ = self._p_hprime_given_h().reshape( (nh,nh,1) )
        p_hprime2_given_h_ = self._p_hprime_given_h().reshape( (nh,1,nh) )
        r_ = self.r(h_prime, h_prime2).reshape( (1,nh,nh) )
        self.results["c_h"] = np.sum( r_ * p_hprime_given_h_ * p_hprime2_given_h_, axis=(1,2) ) * self.dh * self.dh
        return self.results["c_h"]
    
    def c_k(self, c_o_kappa):
        # k, h, kappa are axis-0,1,2, respectively
        # 1/P(k) * \sum_{h,\kappa} g(k|h,\kappa) rho(h) P(\kappa) c_h c_o(\kappa)
        _rho_h = self.rho_h.reshape( (1,self.nh,1) )
        _p_kappa = self.P_kappa.reshape( (1,1,self.nkappa) )
        _c_h = self.c_h().reshape( (1,self.nh,1) )
        return 1.0 / self.P_k() * np.sum( g * _rho_h * _p_kappa * _c_h * c_o_kappa, axis=(1,2) ) * self.dh

        
sampling = NodalSampling(h=h, kappa=kappa, rho_h=rho, P_kappa=P_kappa, r=r)
plt.plot(h, sampling.r_bar_h())
print(sampling.r_bar())


# In[ ]:


g = sampling.g()
print(g.shape)
plt.plot(sampling.k, g[:,300,150])


# In[ ]:


plt.yscale("log")
plt.ylim(1.0e-4,1.0e-1)
plt.xlim(0,50)
plt.plot( sampling.k, sampling.P_k() )


# In[ ]:


plt.plot(h, sampling.r_nn_h(), label="r_nn(h)")
plt.plot(h, sampling.r_bar_h(), label="r(h)")
plt.legend()


# In[ ]:


kappa_mean = np.sum(kappa * P_kappa)
kappa_nn = np.full(kappa.shape, kappa_mean + 1)
k_nn = sampling.k_nn_k(kappa_nn)
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.plot(sampling.k, k_nn)


# In[ ]:


kappa_mean = np.sum(kappa * P_kappa)
kappa_nn = np.full(kappa.shape, kappa_mean + 1)
k_nn = sampling.k_nn_k(kappa_nn)
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.ylim(22,34)
plt.plot(sampling.k, k_nn, label="non assortative")

kappa_nn = np.full(kappa.shape, kappa_mean + 1 + kappa*0.2-30)
#plt.plot(kappa, kappa_nn)
k_nn = sampling.k_nn_k(kappa_nn)
plt.plot(sampling.k, k_nn, label="assortative")

kappa_nn = np.full(kappa.shape, kappa_mean + 1 - kappa*0.2+30)
#plt.plot(kappa, kappa_nn)
k_nn = sampling.k_nn_k(kappa_nn)
plt.plot(sampling.k, k_nn, label="disassortative")
plt.legend(loc="upper left")
plt.savefig("knn_experiment.pdf")


# In[ ]:


plt.plot(sampling.h, sampling.c_h())


# In[ ]:


c_o_kappa = 0.05
plt.yscale("log")
plt.xscale("log")
plt.plot(sampling.k, sampling.c_k(c_o_kappa))


# In[ ]:


c_o = 0.03 # fix the original clustering coefficient
c_o_kappa = np.full(kappa.shape, c_o)
plt.xscale("log")
plt.yscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.plot(sampling.k, sampling.c_k(c_o_kappa), label="const")

c_o_kappa = 1.0 / kappa
c_o_kappa = c_o_kappa / np.sum(c_o_kappa * P_kappa) * 0.03
plt.plot(sampling.k, sampling.c_k(c_o_kappa), label="k^{-1}")
plt.legend(loc="best")

c_o_kappa = 1.0 / (kappa**2)
c_o_kappa = c_o_kappa / np.sum(c_o_kappa * P_kappa) * 0.03
plt.plot(sampling.k, sampling.c_k(c_o_kappa), label="k^{-2}")
plt.legend(loc="best")
plt.savefig("ck_experiment.pdf")


# In[ ]:




