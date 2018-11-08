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
g = nx.random_regular_graph(kappa_0, N, 1234)


# In[ ]:





# In[ ]:


nx.average_neighbor_degree(g)[0]


# In[ ]:


attr = {i:(1.0 if i%2==0 else 0.0) for i in range(N)}
nx.set_node_attributes(g, attr, "h")
g.nodes[6]


# In[ ]:


def calc_homophily(g):
    total = 0
    count = 0
    for e in g.edges:
        total += 1
        if g.nodes[e[0]]['h'] == g.nodes[e[1]]['h']:
            count += 1
    return(count, total)
calc_homophily(g)

def calc_local_homophily(g, i):
    hi = g.nodes[i]['h']
    total = 0
    count = 0
    for j in g[i]:
        hj = g.nodes[j]['h']
        if hi == hj:
            count += 1
        total += 1
    return (count, total)

calc_local_homophily(g,1)


# In[ ]:


# swap attributes of node i and j
def swap_h(g, i, j):
    temp = g.nodes[i]['h']
    g.nodes[i]['h'] = g.nodes[j]['h']
    g.nodes[j]['h'] = temp
swap_h(g, 0, 1)


# In[ ]:


def iterate_swap(g, target_q=0.6, max_iterate=2000, seed = 9876):
    random.seed(seed)
    p,ne = calc_homophily(g)
    target = ne * target_q
    es = list(g.edges)
    for t in range(max_iterate):
        if p == target:
            print(t, "reached target")
            break
        if t % 100 == 0:
            print(t,p,target)
        r = random.randrange(0,len(es))
        n1,n2 = es[r]
        if g.nodes[n1]['h'] == g.nodes[n2]['h']:
            continue
        pn1 = calc_local_homophily(g,n1)[0]
        pn2 = calc_local_homophily(g,n2)[0]
        swap_h(g,n1,n2)
        pn1_n = calc_local_homophily(g,n1)[0]
        pn2_n = calc_local_homophily(g,n2)[0]
        p2 = (pn1_n-pn1) + (pn2_n-pn2)+ p
        #print(p2,p,pn1_n,pn1,pn2_n,pn2)
        #assert( p2 == calc_homophily(g)[0] )
        if abs(p-target) <= abs(p2-target):
            swap_h(g,n1,n2)  # reject the swap
            #print("rejected")
        else:
            #print("accepted")
            p = p2

    
iterate_swap(g,target_q=0.6, max_iterate=200000)
p,ne = calc_homophily(g)
p/ne


# In[ ]:


def run_sampling(g):
    # r(h,h') = h*h'   i.e. sampled when h=h'=1
    g2 = nx.Graph()
    g2.add_nodes_from(range(N))
    new_edges = []
    for e in g.edges:
        if g.nodes[e[0]]['h'] == 1 and g.nodes[e[1]]['h'] == 1:
            new_edges.append(e)
    g2.add_edges_from(new_edges)
    return g2
        
g2 = run_sampling(g)

def plot_Pk(g):
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt)
    
plot_Pk(g2)


# In[ ]:


m = nx.degree_mixing_matrix(g2)
#m
pk = np.sum(m, axis=0)
l = len(pk)
#plt.bar(np.array(range(15)), pk )
np.sum(m*np.array(range(l)).reshape([l,1]), axis=0) / np.sum( m, axis=0)


# In[ ]:




