import sys, collections, math, itertools, numpy, matplotlib, copy, pprint, json, random, time, pickle
from group_formation_experiments import get_rider_graph, generate_weights, nCr, get_relabeled_graph, get_random_subgraph,get_graph_weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import seaborn as sns
matplotlib.use('Agg')
random.seed(111)

import nips_experiment


graphtype           = 'random'
no_MC               =  1
group_size          =  4
rider_sizes         = [8]
number_of_rounds    = 1000 # 6*nCr(rider_sizes[0], group_size) / (rider_sizes[0]/ group_size)

FaceBook = None
Collab = None


MIN_SCORE = 1
MAX_SCORE = 10

number_of_riders = 8
p = math.log(number_of_riders)*1.0/number_of_riders
H = nx.gnp_random_graph(number_of_riders,p)
for i in H.nodes():
    H.node[i]['score'] = random.uniform(MIN_SCORE,MAX_SCORE)

for i in H.edges():
    H[i[0]][i[1]]['color'] = None

#print [x for x in H.nodes(data=True)]

from operator import itemgetter
node_degree_list = sorted(H.degree_iter(),key=itemgetter(1),reverse=True)

def get_incident_edges(edge,H):
    incident_edges = []
    for i in range(2):
        for v in H.neighbors(edge[i]):
            if v != edge[divmod(i+1,2)[1]]:
                incident_edges.append([edge[i],v,H[edge[i]][v]])
    return incident_edges

#print get_incident_edges((0,7),H)
all_colors = range(number_of_riders)

for n,d in node_degree_list:
    edge_list = H.edges(n)
    for e in edge_list:
        if H[e[0]][e[1]]['color'] is None:
            other_colors = set([x['color'] for x in get_incident_edges(e,H)])


def greedy_color(H):

    # len(H)
    # for e in H.edges():
    return None


def get_feedback(H,k_partition,group_size):
    MIN_NOISE = -1.0/group_size
    MAX_NOISE =  1.0/group_size
    feedback = {}
    for group in k_partition:
        temp = np.asarray([x['score'] for x in H.nodes(data=True) if x in group])
        feedback[group] = np.sum(np.outer(temp,temp)) + len(group)*len(group)*random.uniform(MIN_NOISE,MAX_NOISE)
    return feedback # is a dictionary with keys being the groups


def get_k_partition(H,group_size):
    k_partition = []
    partition = []
    for e,x in enumerate(H.nodes()):
        print e,x
        partition.append(x)
        if (e+1)%group_size==0:
            print "current partition:",partition
            k_partition.append(tuple(partition))
            partition = []
    return k_partition

#learning algorithm
m = number_of_riders/group_size
L = 100
epochs = [1]#range(0,group_size+1)

# for epoch in epochs:

#     k_partition = get_k_partition(H,group_size)#is a list of groups. Groups are a list of nodes
#     print k_partition
#     feedback = get_feedback(H,k_partition,group_size)
#     print feedback





