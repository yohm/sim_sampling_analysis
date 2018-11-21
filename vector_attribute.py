#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import entropy


# In[ ]:


hr = np.linspace(0.0, 1.0, num=100)
hr = hr + (hr[1]-hr[0])/2.0
htheta = np.linspace(0.0, 2*math.pi, num=100)
htheta = htheta + (hr[1]-hr[0])/2.0
kappa = np.arange(1,200)

def _rho(hr, htheta):
    # uniform distribution
    nr = hr.shape[0]
    ntheta = htheta.shape[0]
    dr = hr[1]-hr[0]
    dtheta = htheta[1]-htheta[0]
    ro = np.full( (nr,ntheta), 1.0 )  # uniform distribution
    return ro / (np.sum(ro, axis=(0,1))*dr*dtheta)   # normalize

rho = _rho(hr, htheta)
print(rho.shape)
#plt.plot(hr, rho[:,0])
plt.plot(htheta, rho[0,:])


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


def r(r1,theta1,r2,theta2):
    # r1*r2 ( r1*r2-|1-1/pi*|theta2-theta1|| )**2
    delta_theta = np.abs(theta2-theta1)
    a = np.abs( 1.0 - delta_theta / (math.pi) )
    rp = r1*r2
    return rp * (rp - a)**4

for i in range(10):
    plt.plot(htheta, r(i*0.1, htheta, 1.0, 0.0))


# In[ ]:





# In[ ]:


class NodalSampling:
    
    def __init__(self, h1, h2, kappa, rho_h, P_kappa, r):
        self.h1 = h1
        self.nh1 = h1.shape[0]
        self.h2 = h2
        self.nh2 = h2.shape[0]
        self.dh1 = self.h1[1]-self.h1[0]
        self.dh2 = self.h2[1]-self.h2[0]
        
        self.kappa = kappa
        self.nkappa = kappa.shape[0]
        self.rho_h = rho_h
        assert( rho_h.shape == (self.nh1, self.nh2) )
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
            "c_h": None,
            "g_star": None
        }
        
    def r_bar_h(self):
        # sum_{h'} rho(h') r(h,h')
        if self.results["r_bar_h"] is not None:
            return self.results["r_bar_h"]
        # axis (h1,h2, h1', h2')
        h1_prime = np.copy(self.h1).reshape((1,1,self.nh1,1))
        h2_prime = np.copy(self.h2).reshape((1,1,1,self.nh2))
        h1_ = self.h1.reshape( (self.nh1,1,1,1) )
        h2_ = self.h2.reshape( (1,self.nh2,1,1) )
        rhh = self.r(h1_, h2_, h1_prime, h2_prime)
        dr = rhh*self.rho_h.reshape((1,1,self.nh1,self.nh2)) * self.dh1*self.dh2
        self.results["r_bar_h"] = np.sum(dr,axis=(2,3))
        return self.results["r_bar_h"]
    
    def r_bar(self):
        # sum_{h,h'} rho(h') rho(h) r(h,h')
        if self.results["r_bar"] is not None:
            return self.results["r_bar"]
        y = self.r_bar_h() * self.rho_h * self.dh1 * self.dh2
        self.results["r_bar"] = np.sum(y, axis=(0,1))
        return self.results["r_bar"]
    
    def g(self):
        # g(k|h,kappa) = \binom(kappa, k) r(h)^k ( 1-r(h))^{kappa-k}
        #   k,h1,h2,kappa are 0th,1st,2nd,3rd axis, respectively
        if self.results["g"] is not None:
            return self.results["g"]
        _p = self.r_bar_h().reshape( (1,self.nh1,self.nh2,1) )
        _kappa = self.kappa.reshape( (1,1,1,self.nkappa) )
        b = binom(_kappa, _p)
        _k = self.k.reshape( (self.nk,1,1,1) )
        self.results["g"] = b.pmf(_k)
        return self.results["g"]
    
    def P_k(self):
        if self.results["P_k"] is not None:
            return self.results["P_k"]
        _g = g * self.rho_h.reshape( (1,self.nh1,self.nh2,1) ) * self.P_kappa.reshape( (1,1,1,self.nkappa) )
        self.results["P_k"] = np.sum(_g, axis=(1,2,3)) * self.dh1 * self.dh2
        return self.results["P_k"]
    
    def r_nn_h(self):
        # h1,h2,h1_prime,h2_prime are axis=0,1,2,3, respectively.
        if self.results["r_nn_h"] is not None:
            return self.results["r_nn_h"]
        h1_prime = np.copy(self.h1).reshape([1,1,self.nh1,1])
        h2_prime = np.copy(self.h2).reshape([1,1,1,self.nh2])
        rho_h_prime = self.rho_h.reshape([1,1,self.nh1,self.nh2])
        r_bar_h_prime = self.r_bar_h().reshape([1,1,self.nh1,self.nh2])
        r_bar_h = self.r_bar_h().reshape( [self.nh1,self.nh2,1,1] )
        h1_ = self.h1.reshape( [self.nh1,1,1,1] )
        h2_ = self.h2.reshape( [1,self.nh2,1,1] )
        x = r( h1_,h2_,h1_prime,h2_prime ) * rho_h_prime * r_bar_h_prime / r_bar_h
        self.results["r_nn_h"] = np.sum( x, axis=(2,3) ) * self.dh1 * self.dh2
        return self.results["r_nn_h"]
    
    def k_nn_k(self, kappa_nn):
        # k, h1, h2, kappa are axis=0,1,2,3, respectively
        assert( kappa_nn.shape == self.kappa.shape )
        r_nn_h_ = self.r_nn_h().reshape( [1,self.nh1,self.nh2,1] )
        kappa_nn_ = kappa_nn.reshape( [1,1,1,self.nkappa] )
        return 1 + np.sum( self.g_star() * r_nn_h_ * (kappa_nn_-1), axis=(1,2,3) ) * self.dh1 * self.dh2;
    
    def _p_hprime_given_h(self):
        # h1,h2,h1',h2' are axis-0,1,2,3
        # p(h'|h) = r(h',h) rho(h') / r_bar(h)
        if self.results["p_hprime_given_h"] is not None:
            return self.results["p_hprime_given_h"]
        nh1 = self.nh1
        nh2 = self.nh2
        h1_ = self.h1.reshape( (nh1,1,1,1) )
        h2_ = self.h2.reshape( (1,nh2,1,1) )
        h1_prime = self.h1.reshape( (1,1,nh1,1) )
        h2_prime = self.h2.reshape( (1,1,1,nh2) )
        rho_hprime = self.rho_h.reshape( (1,1,nh1,nh2) )
        rbar_h = self.r_bar_h().reshape( (nh1,nh2,1,1) )
        self.results["p_hprime_given_h"] = r(h1_,h2_, h1_prime,h2_prime) * rho_hprime / rbar_h
        return self.results["p_hprime_given_h"]
    
    def c_h(self):
        # h1,h2, h1',h2', h1'',h2'', are axis-0,1,2,3,4,5 respectively
        # \sum_{h', h''} = r(h', h'') * p(h'|h) * p(h''|h)
        if self.results["c_h"] is not None:
            return self.results["c_h"]
        nh1 = self.nh1
        nh2 = self.nh2
        h1_ = self.h1.reshape( (nh1,1,1,1,1,1) )
        h2_ = self.h2.reshape( (1,nh2,1,1,1,1) )
        h1_prime = self.h1.reshape( (1,1,nh1,1,1,1) )
        h2_prime = self.h2.reshape( (1,1,1,nh2,1,1) )
        h1_prime2 = self.h1.reshape( (1,1,1,1,nh1,1) )
        h2_prime2 = self.h2.reshape( (1,1,1,1,1,nh2) )
        p_hprime_given_h_ = self._p_hprime_given_h().reshape( (nh1,nh2,nh1,nh2,1,1) )
        p_hprime2_given_h_ = self._p_hprime_given_h().reshape( (nh1,nh2,1,1,nh1,nh2) )
        r_ = self.r(h1_prime, h2_prime, h1_prime2, h2_prime2).reshape( (1,1,nh1,nh2,nh1,nh2) )
        self.results["c_h"] = np.sum( r_ * p_hprime_given_h_ * p_hprime2_given_h_, axis=(2,3,4,5) ) * (self.dh1*self.dh2)**2
        return self.results["c_h"]
    
    def c_k(self, c_o_kappa):
        # k, h, kappa are axis-0,1,2, respectively
        # 1/P(k) * \sum_{h,\kappa} g(k|h,\kappa) rho(h) P(\kappa) c_h c_o(\kappa)
        _rho_h = self.rho_h.reshape( (1,self.nh,1) )
        _p_kappa = self.P_kappa.reshape( (1,1,self.nkappa) )
        _c_h = self.c_h().reshape( (1,self.nh,1) )
        return 1.0 / self.P_k() * np.sum( g * _rho_h * _p_kappa * _c_h * c_o_kappa, axis=(1,2) ) * self.dh

    def g_star(self):
        # g*(h,kappa|k) = g(k|h,kappa)rho(h)P_o(kappa) / P(k)
        if self.results["g_star"] is not None:
            return self.results["g_star"]
        Pk = self.P_k()
        Pk_ = Pk[ Pk > 0 ]
        Pk_ = Pk_.reshape( [Pk_.shape[0],1,1,1,] )
        _g = g[Pk > 0,:,:]
        rho_h_ = self.rho_h.reshape( [1,self.nh1,self.nh2,1] )
        p_kappa_ = self.P_kappa.reshape( [1,1,self.nkappa] )
        self.results["g_star"] = _g / Pk_ * rho_h_ * p_kappa_
        return self.results["g_star"]


# In[ ]:


sampling = NodalSampling(h1=hr, h2=htheta, kappa=kappa, rho_h=rho, P_kappa=P_kappa, r=r)
plt.plot(hr, sampling.r_bar_h()[:,0] )
print(sampling.r_bar())


# In[ ]:


g = sampling.g()
print(g.shape)
plt.plot(sampling.k, g[:,80,0,70])


# In[ ]:


plt.yscale("log")
plt.ylim(1.0e-4,1.0e0)
plt.xlim(0,50)
plt.plot( sampling.k, sampling.P_k() )


# In[ ]:


plt.plot(hr, sampling.r_nn_h()[:,0], label="r_nn(h)")
plt.plot(hr, sampling.r_bar_h()[:,0], label="r(h)")
plt.legend()


# In[ ]:


kappa_mean = np.sum(kappa * P_kappa)
kappa_nn = np.full(kappa.shape, kappa_mean + 1)
k_nn = sampling.k_nn_k(kappa_nn)
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.plot(sampling.k, k_nn)
k_nn


# In[ ]:


kappa_mean = np.sum(kappa * P_kappa)
kappa_nn = np.full(kappa.shape, kappa_mean + 1)
k_nn = sampling.k_nn_k(kappa_nn)
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.plot(sampling.k, k_nn, label="non assortative")

kappa_nn = np.full(kappa.shape, kappa_mean + 1 + kappa*0.2-30)
k_nn = sampling.k_nn_k(kappa_nn)
plt.plot(sampling.k, k_nn, label="assortative")

kappa_nn = np.full(kappa.shape, kappa_mean + 1 - kappa*0.2+30)
k_nn = sampling.k_nn_k(kappa_nn)
plt.plot(sampling.k, k_nn, label="disassortative")
plt.legend(loc="upper left")
plt.savefig("knn_experiment.pdf")


# In[ ]:


plt.plot(sampling.h1, sampling.c_h()[:,0])


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


g_star_k_h = np.sum(sampling.g_star(), axis=2)
#plt.plot(h, g_star_k_h[10,:])
#plt.plot(h, rho)
kls = [entropy( g_star_k_h[k,:], rho ) for k in range( g_star_k_h.shape[0] )]
plt.plot(sampling.k, kls)


# In[ ]:


g_star_k_kappa = np.sum(sampling.g_star(), axis=1) * (h[1]-h[0])
plt.plot(kappa, g_star_k_kappa[10,:], label=r"$k = 10$")
plt.plot(kappa, g_star_k_kappa[50,:], label=r"$k = 50$")
plt.plot(kappa, P_kappa, label =r"$P(\kappa)$")
plt.legend()


# In[ ]:


#kls = [entropy( P_kappa, g_star_k_kappa[k,:] ) for k in range( g_star_k_h.shape[0] )]
kls = [entropy( g_star_k_kappa[k,:], P_kappa ) for k in range( g_star_k_h.shape[0] )]
plt.xscale("log")
plt.xlim(1.0e0, 1.0e2)
plt.ylim(0,10)
plt.plot(sampling.k, kls)
plt.savefig("kl_divergence.pdf")


# In[ ]:




