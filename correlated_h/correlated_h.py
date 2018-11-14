#!/usr/bin/env python
# coding: utf-8

import sys
import networkx as nx
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import numpy as np
import json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class CorrelatedH():

    @classmethod
    def set_h_weibull(cls, g, alpha, h_0, seed=2345):
        np.random.seed(seed)
        hs = []
        while True:
            if len(hs) == N:
                break
            h = np.random.weibull(alpha, N-len(hs)) * h_0
            hs += h[h<1.0].tolist()
        attr = {i:x for i,x in enumerate(hs) }
        nx.set_node_attributes(g, attr, "h")

    @classmethod
    def calc_hdiff(cls, g):
        total = 0
        d = 0.0
        for e in g.edges:
            total += 1
            d += abs(g.nodes[e[0]]['h'] - g.nodes[e[1]]['h'])
        return(d, total)

    @classmethod
    def calc_local_hdiff(cls, g, i):
        hi = g.nodes[i]['h']
        total = 0
        d = 0.0
        for j in g[i]:
            hj = g.nodes[j]['h']
            d += abs(hi - hj)
            total += 1
        return (d, total)

    @classmethod
    def hdiff(cls, g):
        d,total = cls.calc_hdiff(g)
        return d / total

    @classmethod
    def swap_h(cls, g, i, j):
        temp = g.nodes[i]['h']
        g.nodes[i]['h'] = g.nodes[j]['h']
        g.nodes[j]['h'] = temp

    @classmethod
    def iterate_swap(cls, g, target_hdiff=0, max_iterate=1000, seed = 9876, figfile="timeseries.png"):
        _g = g.copy()
        plot_x = []
        plot_y = []
        random.seed(seed)
        q,ne = cls.calc_hdiff(_g)
        target = ne * target_hdiff
        es = list(_g.edges)
        for t in range(max_iterate):
            if q == target:
                eprint(t, "reached target")
                break
            if t % 100 == 0:
                plot_x.append(t)
                plot_y.append(q/ne)
            if t % (max_iterate/100) == 0:
                eprint(t,q,target)
            #r = random.randrange(0,len(es))
            #n1,n2 = es[r]
            n1 = random.randrange(0, N)
            n2 = random.randrange(0, N)
            if _g.nodes[n1]['h'] == _g.nodes[n2]['h']:
                continue
            pn1 = cls.calc_local_hdiff(_g,n1)[0]
            pn2 = cls.calc_local_hdiff(_g,n2)[0]
            cls.swap_h(_g,n1,n2)
            pn1_n = cls.calc_local_hdiff(_g,n1)[0]
            pn2_n = cls.calc_local_hdiff(_g,n2)[0]
            q2 = (pn1_n-pn1) + (pn2_n-pn2) + q
            #assert( p2 == calc_homophily(g)[0] )
            if abs(q-target) <= abs(q2-target):
                cls.swap_h(_g,n1,n2)  # reject the swap
            else:
                q = q2
        if figfile:
            plt.xlabel("t")
            plt.ylabel(r"$\Delta h$")
            plt.plot(plot_x, plot_y, label="actual")
            plt.savefig(figfile)
            plt.clf()
        return _g

class Sampling():

    @classmethod
    def gen_mean(cls,x,y,beta):  # when beta<=-10 or >=10, minimum or maximum is used
        if beta == 0:
            return np.sqrt(x*y)
        elif beta <= -10:
            return np.minimum(x,y)
        elif beta >= 10:
            return np.maximum(x,y)
        else:
            return ((x**beta+y**beta)/2)**(1.0/beta)

    @classmethod
    def run_sampling(cls, g, beta, seed=9742):
        r = lambda x, y: cls.gen_mean(x,y,beta)
        random.seed(seed)
        g2 = nx.classes.function.create_empty_copy(g)
        #g2 = nx.Graph()
        #g2.add_nodes_from(range(len(g)))
        new_edges = []
        for e in g.edges:
            h1 = g.nodes[e[0]]['h']
            h2 = g.nodes[e[1]]['h']
            if random.random() < r(h1,h2):
                new_edges.append(e)
        g2.add_edges_from(new_edges)
        return g2

params = {"N": 5000, "kappa_0": 150, "alpha": 0.8, "h_0": 0.1, "beta":-10, "iteration": 10000, "_seed": 1234567890}
if len(sys.argv) != 2:
    eprint("Usage: python correlated.py _input.json")
    sys.exit(1)
with open(sys.argv[1]) as f:
    params = json.load(f)

eprint("params", params)

N = params["N"]
kappa_0 = params["kappa_0"]
alpha = params["alpha"]
h_0 = params["h_0"]
iteration = params["iteration"]
s0 = params["_seed"]
beta = params["beta"]

g = nx.erdos_renyi_graph(N, kappa_0/N, seed=s0+1234)

CorrelatedH.set_h_weibull(g, alpha, h_0, seed=s0+2345)
g2 = CorrelatedH.iterate_swap(g, target_hdiff=0.0, max_iterate=iteration, seed=s0+3456)
eprint("Dh: ", CorrelatedH.hdiff(g), "->", CorrelatedH.hdiff(g2) )

g3 = Sampling.run_sampling(g2, beta, seed=s0+3456)

def plot_Pk(g, filename="pk.png"):
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip( *sorted(degreeCount.items()))
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.plot(deg, cnt)
    plt.savefig(filename)
    plt.clf()
    return list(zip(deg,cnt))

    
pk = plot_Pk(g3)
np.savetxt("pk.txt", pk)

def plot_knn(g, filename="knn.png"):
    knn = nx.k_nearest_neighbors(g)
    lists = sorted(knn.items())
    x, y = zip(*lists)
    plt.xlabel("k")
    plt.ylabel(r"$k_{nn}(k)$")
    plt.plot(x,y)
    plt.savefig(filename)
    plt.clf()
    return list(zip(x,y))

knn = plot_knn(g3)
np.savetxt("knn.txt", knn)

def plot_ck(g, filename="ck.png"):
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
    plt.xlabel("k")
    plt.ylabel(r"$c(k)$")
    plt.plot(x,y)
    plt.savefig(filename)
    plt.clf()
    return list(zip(x,y))
    
ck = plot_ck(g3)
np.savetxt("ck.txt", ck)

def average_k(g):
    return np.sum( [k for i,k in nx.degree(g)] ) / len(g)

with open("_output.json", "w") as f:
    j = {"average_k": average_k(g3), "average_c": nx.average_clustering(g3), "hdiff_before_sampling": CorrelatedH.hdiff(g2), "hdiff_after_sampling": CorrelatedH.hdiff(g3)}
    json.dump(j, f, indent=4)
    f.flush()

