import sys, collections, math, itertools, numpy, matplotlib, copy, pprint, json, random, time, pickle
from group_formation_experiments import get_rider_graph, generate_weights, nCr, get_relabeled_graph, get_random_subgraph,get_graph_weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import seaborn as sns
matplotlib.use('Agg')
random.seed(111)

import nips_experiment


no_MC               =  10
group_size          =  4
number_of_riders    =  16
m = number_of_riders/group_size
L = 100

MIN_SCORE = 1
MAX_SCORE = 10
p = math.log(number_of_riders)*1.0/number_of_riders
H = nx.gnp_random_graph(number_of_riders,p)
for i in H.nodes():
    H.node[i]['score'] = random.uniform(MIN_SCORE,MAX_SCORE)

for i in H.nodes():
    print H.node[i]['score']

def get_feedback(H,k_partition,group_size):
    MIN_NOISE = -1.0/group_size
    MAX_NOISE =  1.0/group_size
    feedback = {}
    for group in k_partition:
        temp = numpy.asarray([H.node[x]['score'] for x in H.nodes() if x in group])
        # print 'temp',temp
        feedback[group[-1]] = np.sum(np.outer(temp,temp)) + 1000.0*random.uniform(MIN_NOISE,MAX_NOISE)
        # print "the feedback is",feedback[group[-1]]
    return feedback # is a dictionary with keys being the groups


def do_stuff(H,L,group_size,number_of_riders):

    partitions = {}
    feedbacks = {}
    for e in H.edges():
        rest = []
        for x in H.nodes(): 
            if x not in e:
                rest.append(x)
        temp = rest[:group_size-1]
        temp2 = copy.deepcopy(temp)
        # print temp
        temp.append(e[0])
        temp2.append(e[1])
        partitions[e] = [temp,temp2]
        feedbacks[e] = []
        # print partitions[e]
        for i in range(L):
            feedbacks[e].append(get_feedback(H,partitions[e],group_size))

    pprint.pprint(partitions)

    score_diff = {}
    for e in H.edges():
        batch = {}
        for i in range(2):
            temp = 0
            for sample in feedbacks[e]:
                temp += sample[e[i]]
            batch[e[i]] = max(0,temp)*1.0/L
        # print "batch",batch
        score_diff[e] = math.sqrt(batch[e[0]]) - math.sqrt(batch[e[1]])

    pprint.pprint(score_diff)

    from operator import itemgetter
    node_degree_list = sorted(H.degree_iter(),key=itemgetter(1),reverse=True)
    print node_degree_list[0][0]

    paths = nx.shortest_path(H,source=node_degree_list[0][0])
    pprint.pprint(paths)

    top_node = node_degree_list[0][0]
    final_scores = {}
    for v in paths:
        final_scores[v] = 0
        if len(paths[v]) == 1:
            final_scores[v] = 0
        elif len(paths[v]) ==2:
            if (top_node,v) in score_diff:
                final_scores[v] = score_diff[(top_node,v)]
            else:
                final_scores[v] = score_diff[(v,top_node)]
        else:
            for i,x in enumerate(paths[v][:-1]):
                #print 'indexing',(paths[v][i],paths[v][i+1]) 
                if (paths[v][i],paths[v][i+1]) in score_diff:
                    final_scores[v] += score_diff[(paths[v][i],paths[v][i+1])]
                else:
                    final_scores[v] -= score_diff[(paths[v][i+1],paths[v][i])]


    true_scores = []
    estimated_scores = []
    for v in H.nodes():
        true_score = -H.node[top_node]['score'] + H.node[v]['score']
        estimated_score = final_scores[v]
        true_scores.append(abs(true_score))
        estimated_scores.append(abs(estimated_score))

    error_magnitude = numpy.linalg.norm(numpy.asarray(true_scores) - numpy.asarray(estimated_scores))/numpy.linalg.norm(true_scores)
    print error_magnitude

    return error_magnitude

performance = []
for round_number in range(no_MC):
    temp = []
    for lval in [10,100,200,500,1000]:
        temp.append(do_stuff(H,lval,group_size,number_of_riders))
    performance.append(temp)

print performance


fig = plt.figure()
ax = fig.add_subplot(111)

temp = numpy.asarray(performance)
#print temp
temp_avg = numpy.mean(temp, axis=0)
#print temp_avg
temp_std = numpy.std(temp, axis=0)

xs = [10,100,200,500,1000]
ys = temp_avg

ax.fill_between(xs, ys-temp_std, ys+temp_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
ax.plot(xs, ys)

plt.show()
plt.xlabel('Sample size #')
plt.ylabel('Normalized rror')
plt.title('Error between estimates and true scores')
plt.tight_layout()
plt.ylim((0,1.2*max(ys)))

plot_dir = '.'
fname = plot_dir + '/scores_fig' + '.png'
print "plot will be saved at:",fname
plt.savefig(fname)