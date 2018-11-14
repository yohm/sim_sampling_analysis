#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import random
import matplotlib.pyplot as plt
import collections
import numpy as np


# In[ ]:


N = 5000
kappa_0 = 150
#g = nx.random_regular_graph(kappa_0, N, 1234)
g = nx.erdos_renyi_graph(N, kappa_0/N, seed=1234)
len(g)


# In[ ]:


def rho_h(N, alpha, h_0, seed=3420):
    np.random.seed(seed)
    hs = []
    while True:
        if len(hs) == N:
            return hs
        h = np.random.weibull(alpha, N-len(hs)) * h_0
        #print(h[h<0.2].shape)
        hs += h[h<1.0].tolist()

attr = {i:x for i,x in enumerate( rho_h(N,0.8,0.1) ) }
#attr = {i:(1.0 if i%2==0 else 0.0) for i in range(N)}
nx.set_node_attributes(g, attr, "h")
g.nodes[6]


# In[ ]:


def calc_hdiff(g):
    total = 0
    d = 0.0
    for e in g.edges:
        total += 1
        d += abs(g.nodes[e[0]]['h'] - g.nodes[e[1]]['h'])
    return(d, total)
d,total = calc_hdiff(g)
print(d/total)

def calc_local_hdiff(g, i):
    hi = g.nodes[i]['h']
    total = 0
    d = 0.0
    for j in g[i]:
        hj = g.nodes[j]['h']
        d += abs(hi - hj)
        total += 1
    return (d, total)

print(calc_local_hdiff(g,1))


# In[ ]:


# swap attributes of node i and j
def swap_h(g, i, j):
    temp = g.nodes[i]['h']
    g.nodes[i]['h'] = g.nodes[j]['h']
    g.nodes[j]['h'] = temp
swap_h(g, 0, 1)


# In[ ]:


def iterate_swap(g, target_q=0.6, max_iterate=1000, seed = 9876):
    _g = g.copy()
    plot_x = []
    plot_y = []
    plot_y2 = []
    random.seed(seed)
    q,ne = calc_hdiff(_g)
    target = ne * target_q
    es = list(_g.edges)
    for t in range(max_iterate):
        if q == target:
            print(t, "reached target")
            break
        if t % 100 == 0:
            plot_x.append(t)
            plot_y.append(q)
            plot_y2.append(target)
        if t % (max_iterate/100) == 0:
            print(t,q,target)
        #r = random.randrange(0,len(es))
        #n1,n2 = es[r]
        n1 = random.randrange(0, N)
        n2 = random.randrange(0, N)
        if _g.nodes[n1]['h'] == _g.nodes[n2]['h']:
            continue
        pn1 = calc_local_hdiff(_g,n1)[0]
        pn2 = calc_local_hdiff(_g,n2)[0]
        swap_h(_g,n1,n2)
        pn1_n = calc_local_hdiff(_g,n1)[0]
        pn2_n = calc_local_hdiff(_g,n2)[0]
        q2 = (pn1_n-pn1) + (pn2_n-pn2) + q
        #print(p2,p,pn1_n,pn1,pn2_n,pn2)
        #assert( p2 == calc_homophily(g)[0] )
        if abs(q-target) <= abs(q2-target):
            swap_h(_g,n1,n2)  # reject the swap
            #print("rejected")
        else:
            #print("accepted")
            q = q2
    plt.plot(plot_x, plot_y, label="actual")
    plt.plot(plot_x, plot_y2, label="target")
    return _g

    
g_swap = iterate_swap(g,target_q=0.1, max_iterate=100000)
p,ne = calc_hdiff(g_swap)
print(p/ne)
p,ne = calc_hdiff(g)
print(p/ne)


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


def run_sampling(g, seed = 3456):
    random.seed(seed)
    # r(h,h') = h+h'/2   i.e. h = (0.6, 0.4)
    g2 = nx.Graph()
    g2.add_nodes_from(range(N))
    new_edges = []
    for e in g.edges:
        h1 = g.nodes[e[0]]['h']
        h2 = g.nodes[e[1]]['h']
        if random.random() < r(h1,h2):
            new_edges.append(e)
    g2.add_edges_from(new_edges)
    return g2
        
g2 = run_sampling(g)
g2_swap = run_sampling(g_swap)

def plot_Pk(g):
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt)
    
plot_Pk(g2)
plot_Pk(g2_swap)


# In[ ]:


def plot_knn(g):
    knn = nx.k_nearest_neighbors(g)
    lists = sorted(knn.items())
    x, y = zip(*lists)
    plt.plot(x,y)

plot_knn(g2)
plot_knn(g2_swap)


# In[ ]:


def plot_ck(g):
    ks = [v for k,v in nx.degree(g)]
    cs = [v for k,v in nx.clustering(g).items()]
    ck_sum = {}
    ck_count = {}
    for k,c in zip(ks, cs):
        ck_sum[k] = ck_sum.get(k,0.0) + c
        ck_count[k] = ck_count.get(k,0) + 1
    ck = {}
    for k,c_sum in ck_sum.items():
        ck[k] = ck_sum[k] / ck_count[k]
    x,y = zip(*sorted(ck.items()))
    plt.plot(x,y)
    
plot_ck(g2)
plot_ck(g2_swap)


# In[ ]:


np.sum( [k for i,k in nx.degree(g2_swap)] ) / len(g2_swap)


# In[ ]:


nx.average_clustering(g2_swap)


# In[ ]:


nx.classes.function.create_empty_copy(g2_swap)


# In[ ]:




