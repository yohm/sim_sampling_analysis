import networkx as nx

net = nx.erdos_renyi_graph(1000, 0.05)
for (i,j) in net.edges():
    print( '%d %d 1'%(i,j) )
