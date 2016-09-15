import time
print "Import start: {0} ".format(time.ctime())
import math, numpy, matplotlib, copy, pprint, random
matplotlib.use('Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import networkx as nx
random.seed(111)
print "Import end: {0} ".format(time.ctime())

def get_feedback(H,k_partition,group_size):
    MIN_NOISE = -1.0/group_size
    MAX_NOISE =  1.0/group_size
    feedback = {}
    for group in k_partition:
        temp = numpy.asarray([H.node[x]['score'] for x in H.nodes() if x in group])
        # print 'temp',temp
        feedback[group[-1]] = numpy.sum(numpy.outer(temp,temp)) + 1000.0*random.uniform(MIN_NOISE,MAX_NOISE)
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

    # pprint.pprint(partitions)

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

    # pprint.pprint(score_diff)

    from operator import itemgetter
    node_degree_list = sorted(H.degree_iter(),key=itemgetter(1),reverse=True)
    # print node_degree_list[0][0]

    paths = nx.shortest_path(H,source=node_degree_list[0][0])
    # pprint.pprint(paths)

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
    # print error_magnitude

    return error_magnitude


def get_graph(number_of_riders,MIN_SCORE,MAX_SCORE):

    print "Generating Gnp graph start: {0} ".format(time.ctime())
    p = math.log(number_of_riders)*1.0/number_of_riders
    H = nx.gnp_random_graph(number_of_riders,p)
    for i in H.nodes():
        H.node[i]['score'] = random.uniform(MIN_SCORE,MAX_SCORE)

    print "Generating Gnp graph end: {0} ".format(time.ctime())

    # for i in H.nodes():
    #     print H.node[i]['score']
    return H


def get_plot(performance,lvalArray,graph_type):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    temp = numpy.asarray(performance)
    #print temp
    temp_avg = numpy.mean(temp, axis=0)
    #print temp_avg
    temp_std = numpy.std(temp, axis=0)

    xs = lvalArray
    ys = temp_avg

    ax.fill_between(xs, ys-temp_std, ys+temp_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    ax.plot(xs, ys)

    plt.show()
    plt.xlabel('t')
    plt.ylabel('Error')
    # plt.title('Error between estimates and true scores')
    # plt.gca.tight_layout()
    plt.ylim((0,1.2*max(ys)))

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)


    plot_dir = '.'
    fname = plot_dir + '/learning_order_' + graph_type+'_16.png'
    print "plot will be saved at:",fname
    plt.savefig(fname)



def main():
    print "Main start: {0} ".format(time.ctime())

    no_MC               =   30 # 1 #
    group_size          =   4
    number_of_riders    =  16
    m = number_of_riders/group_size
    L = 100
    graph_type = 'facebook' # 'random' #

    MIN_SCORE = 1
    MAX_SCORE = 10


    if graph_type=='facebook':
        #Load original graph
        print "Loading facebook graph from pickle..."
        start_time_fb = time.time()
        temp_dir = 'D:/usb2/datasets/group_formation_temp'
        GraphInput = nx.read_gpickle(temp_dir+"/fbgraph.gpickle")
        print "Facebook graph loaded in sec:",time.time() - start_time_fb


        nodes = GraphInput.nodes()
        subgraph_nodes = random.sample(nodes, number_of_riders)
        H1 = GraphInput.subgraph(subgraph_nodes)

        label_mapping = {}
        nodes = H1.nodes()
        for index in range(len(nodes)):
            label_mapping[nodes[index]] = int(index)
        H = nx.relabel_nodes(H1, label_mapping)

        for i in H.nodes():
            H.node[i]['score'] = random.uniform(MIN_SCORE,MAX_SCORE)

    elif graph_type=='random':
        H = get_graph(number_of_riders,MIN_SCORE,MAX_SCORE)
    else:
        print "ERROR"

    lvalArray = [10,40,80,120,160,200,250,300,350,400,500,750,1000]

    performance = []
    for round_number in range(no_MC):
        print 'MC run: {0}. Time : {1}'.format(round_number,time.ctime())
        temp = []
        for lval in lvalArray:
            temp.append(do_stuff(H,lval,group_size,number_of_riders))
        performance.append(temp)

    # print performance

    get_plot(performance,lvalArray,graph_type)


if __name__=="__main__":
    main()