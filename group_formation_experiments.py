import random

import networkx as nx
import time
import pulp
import itertools
import collections
import numpy as np
import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

__author__ = 'q4BKGR0Q,q4fj4lj9'


random.seed(1000)  # for replicability of experiments.


#---------------------- GREEDY BEG ---------------------------
def get_coalesced_graph(graph, matching, version='min'):
    H = nx.Graph()
    for u, v in matching.iteritems():
        temp = [str(u), str(v)] #THIS SEEMS CRITICAL. Does greedy change labels to strings or expect labels being strings?
        temp.sort()
        H.add_node(temp[0] + '_' + temp[1])
    for u in H.nodes():
        for v in H.nodes():
            if u == v:
                continue
            ##debug
            # print "u",u
            # print "v",v
            weight = get_edge_weight(graph, u, v, version)
            H.add_edge(u, v, weight=weight)
    return H

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_edge_weight(graph, u, v, version='min'):

    # #debug
    # for line in nx.generate_edgelist(graph, data=True):
        # print(line)

    u_indices = find(u, '_')
    v_indices = find(v, '_')

    ##debug
    # print u_indices
    # print v_indices

    u_mid = u_indices[len(u_indices) / 2]
    v_mid = v_indices[len(v_indices) / 2]
    u_1 = u[:u_mid]
    u_2 = u[u_mid + 1:]
    v_1 = v[:v_mid]
    v_2 = v[v_mid + 1:]

    #debug
    # print "debug:" ,u_1,v_1
    # print graph[u_1][v_1]['weight']

    if version == 'min':
        weight = min(graph.get_edge_data(u_1, v_1)['weight'], graph.get_edge_data(u_1, v_2)['weight'],
                     graph.get_edge_data(u_2, v_1)['weight'], graph.get_edge_data(u_2, v_2)['weight'])
    elif version == 'avg':
        weight = sum([graph.get_edge_data(u_1, v_1)['weight'], graph.get_edge_data(u_1, v_2)['weight'],
                      graph.get_edge_data(u_2, v_1)['weight'], graph.get_edge_data(u_2, v_2)['weight']]) * 1.0 / 4
    return weight

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def solve_greedy(G, group_size, MAX_WEIGHT,version='min'):
    """
    Greedy maximum weight matching in the graph of riders, using networkx graph library
    """

    # check version. TBD: use assertion
    if version != 'min' and version != 'avg':
        print "ILP: Error incorrect option specified. Please specify version correctly as min or avg."
        return -1


    #Theja fix: make node ids strings
    if type(G.nodes()[1])==int:
        G0 = G.copy()
        G = nx.Graph()
        for u,v,x in G0.edges(data=True):
            G.add_edge(str(u),str(v),x)

        #debug
        # pprint.pprint(G.edges(data=True))
        # pprint.pprint(G.edges(data=True))

    start_time = time.time()
    current_group_size = 1
    original_graph = G
    coalesced_graph = None
    while current_group_size < group_size:
        # print current_group_size
        graph_matching = nx.max_weight_matching(original_graph)

        #debug
        # print [x for x in graph_matching.iteritems()]
        # for line in nx.generate_edgelist(original_graph, data=True):
        #     print(line)

        coalesced_graph = get_coalesced_graph(original_graph, graph_matching, version)
        # print "debug:coalesced_graph:",coalesced_graph.nodes()
        # print "debug:coalesced_graph:",coalesced_graph.edges(data=True)
        original_graph = coalesced_graph
        current_group_size *= 2

    #debug
    #print "Greedy groups:", coalesced_graph.nodes()

    groups = []
    for n in coalesced_graph.nodes():
        groups.append([int(x) for x in n.split('_')])

    #debug
    #print "Greedy groups:",groups


    smallest_w = [MAX_WEIGHT] * len(coalesced_graph.nodes())
    avg_w = [0] * len(coalesced_graph.nodes())
    for i, u in enumerate(coalesced_graph.nodes()):
        group_elements = str(u).split('_')
        for v, w in itertools.combinations(group_elements, 2):
            # print "Edge weight between {0} and {1} is {2}".format(str(v),str(w),G.get_edge_data(v,w)['weight'])
            if smallest_w[i] > G.get_edge_data(v, w)['weight']:
                smallest_w[i] = G.get_edge_data(v, w)['weight']
            avg_w[i] += G.get_edge_data(v, w)['weight']
        avg_w[i] = (1.0 * avg_w[i]) / group_size

    time_taken = time.time() - start_time
    #debug
    # if version == 'min':
    #     print "Greedy groupwise smallest happiness:", smallest_w
    #     print "Greedy Optimal objective value: {0}".format(min(smallest_w))
    # if version == 'avg':
    #     print "Greedy groupwise average happiness:", avg_w
    #     print "Greedy Optimal objective value: {0}".format(min(avg_w))
    # print 'Greedy finished in ', (time_taken)
    
    if version=='min':
        return {'obj':min(smallest_w),'time':time_taken,'number_of_riders':len(G.nodes()), 'groups':groups}
    if version=='avg':
        return {'obj':min(avg_w)*1.0*(nCr(group_size, 2)/group_size),'time':time_taken,'number_of_riders':len(G.nodes()), 'groups':groups}

    return {'obj': min(smallest_w), 'time': time_taken, 'number_of_riders': len(G.nodes()), 'groups':groups}
#---------------------- GREEDY END ---------------------------

#---------------------- ILP BEG ---------------------------

def solve_ILP(N, K, weights, MAX_WEIGHT, DEFAULT_WEIGHT, version="min",temp_dir="."):
    '''
    This ILP solves the group formation problem
    '''

    start_time = time.time()

    # check version
    if version != 'min' and version != 'avg':
        print "ILP: Error incorrect option specified. Please specify version correctly as min or avg."
        return -1

    # Setting up the MILP
    prob = pulp.LpProblem('offline grouping', pulp.LpMaximize)

    # Decision variable Indices
    def initIndicesIJK(N, K):
        ijk = []
        for i in range(N):
            for j in range(N):
                if i < j:
                    for k in range(K):
                        ijk.append((i, j, k))
        return ijk

    def initIndicesIK(N, K):
        ik = []
        for i in range(N):
            for k in range(K):
                ik.append((i, k))
        return ik

    def initIndicesIJ(N):
        ij = []
        for i in range(N):
            for j in range(N):
                if i < j:
                    ij.append((i, j))
        return ij

    ijkIndices = initIndicesIJK(N, K)
    ikIndices = initIndicesIK(N, K)
    # print "ILP: indices created"
    # print "ILP: ijkIndices ... (size={0})".format(len(ijkIndices))
    # print "ILP: ikIndices ... (size={0})".format(len(ikIndices))
    if version == 'avg':
        ijIndices = initIndicesIJ(N)
        # print "ILP: ijIndices ... (size={0})".format(len(ijIndices))


    # Decision Variables
    xijk_vars = pulp.LpVariable.dicts("x",
                                      [idx for idx in ijkIndices],
                                      0,
                                      1,
                                      pulp.LpInteger)
    yik_vars = pulp.LpVariable.dicts("y",
                                     [idx for idx in ikIndices],
                                     0,
                                     1,
                                     pulp.LpInteger)
    ck_vars = pulp.LpVariable.dicts("c",
                                    [k for k in range(K)],
                                    0,
                                    MAX_WEIGHT * N,
                                    pulp.LpContinuous)
    cmin_var = pulp.LpVariable("cmin",
                               0,
                               MAX_WEIGHT * N,
                               pulp.LpContinuous)

    # print "ILP: Decision variables initiated"

    # Objective
    prob += cmin_var  # - (1.0/(N*K))*pulp.lpSum([xijk_vars[(i,j,k)] for i,j,k in ijkIndices])

    # Constraints
    # cmin_var is smaller than each group's ck_var value
    for k in range(K):
        prob += cmin_var <= ck_vars[k]
    # For each pair of vertices, (1) c_k should be smaller than weight if that
    # pair is in group k (2) for every i,j,k xijk should be smaller than or
    # equal to yik (3) for every i,j,k xijk should be bigger than yik + yjk -1
    # [KOYEL's idea]

    if version == "avg":
        for k in range(K):
            prob += ck_vars[k] == (1.0 * K / N) * pulp.lpSum(
                [weights[(i, j)] * xijk_vars[(i, j, k)] for i, j in ijIndices])

    for i, j, k in ijkIndices:
        if version == "min":
            prob += ck_vars[k] <= DEFAULT_WEIGHT * (1 - xijk_vars[(i, j, k)]) + weights[(i, j)] * xijk_vars[(i, j, k)]
        prob += xijk_vars[(i, j, k)] <= yik_vars[(i, k)]
        prob += xijk_vars[(i, j, k)] <= yik_vars[(j, k)]
        prob += xijk_vars[(i, j, k)] >= yik_vars[(i, k)] + yik_vars[(j, k)] - 1

    # For every vertex j, it should only belong to one group
    for j in range(N):
        prob += pulp.lpSum([yik_vars[(i, k)] for i, k in ikIndices if i == j]) == 1
    # size of the group should be N/K
    for kprime in range(K):
        prob += pulp.lpSum([yik_vars[(i, k)] for i, k in ikIndices if k == kprime]) == N / K
    # For every pair of vertices i,j, they should only belong to atmost one group
    for i_fixed in range(N):
        for j_fixed in range(N):
            prob += pulp.lpSum([xijk_vars[(i, j, k)] for i, j, k in ijkIndices if i == i_fixed and j == j_fixed]) <= 1
    # #Symmetry of Xijk vars
    # for i in range(N):
    #     for j in range(N):
    #         if j !=i:
    #             for k in range(K):
    #                 prob += xijk_vars[(i,j,k)] == xijk_vars[(j,i,k)]

    print "ILP: All constraints generated."

    #need this on cplex windows otherwise getting a key error _dummy
    prob.writeLP(temp_dir+'/group_formation_ilp.lp')
    print "ILP written at "+ temp_dir+"/group_formation_ilp.lp"

    # Solving the MILP
    try:
        status = prob.solve(pulp.CPLEX(timelimit=180))
    except pulp.PulpSolverError:
        status = prob.solve()

    # print "ILP: Solver status: {0}".format(pulp.LpStatus[status])

    print "ILP: Optimal objective value (max(min_[k in groups](c_k))): {0}".format(cmin_var.varValue)

    groups = []

    for k in range(K):
        print "ILP: Group elements (derived from Yik) with ", version, " weight :", ck_vars[k].varValue, [i for i in range(N) if yik_vars[(i,k)].varValue == 1]
        groups.append([i for i in range(N) if yik_vars[(i, k)].varValue == 1])

    for k in range(K):
        # print "ILP: Xijk vars for Group {0} with ", version, " weight {1}:".format(k, ck_vars[k].varValue)
        wMat = np.zeros((N / K, N / K))
        rowIndex = 0
        nodeIndex = []
        for i in range(N):
            if yik_vars[(i, k)].varValue == 1:
                nodeIndex.append(i)
        rowIndex = 0
        for u in nodeIndex:
            colIndex = 0
            for v in nodeIndex:
                wMat[rowIndex, colIndex] = weights[(u, v)]
                colIndex += 1
            rowIndex += 1
        # print "ILP: wMat ..."
        pprint.pprint(wMat - np.eye(N / K) * DEFAULT_WEIGHT)

    if version=="avg":
        for k in range(K):
            print "group ",k
            print "normalization",1.0*K/N
            # for e in ijIndices:
            #     (i,j) = e
            #     print e, weights[(i,j)],xijk_vars[(i,j,k)].varValue

    time_taken = time.time() - start_time
    print 'ILP finished in ', (time_taken)
    # for kp in range(K):
    #     for i,j,k in ijkIndices:
    #         if k==kp and xijk_vars[(i,j,k)].varValue==1:
    #             print "X_({0},{1},{2}) = {3}".format(i,j,k,xijk_vars[(i,j,k)].varValue)

    return {'obj': cmin_var.varValue, 'time': time_taken, 'number_of_riders': N, 'groups': groups}

#---------------------- ILP END ---------------------------

#---------------------- HELPER FUNCTIONS ---------------------------

# function to generate a graph given the number of vertices
def get_rider_graph(number_of_riders, weights):
    """
    Generate a rider graph
    """
    # populating the rider graph. 
    # Even though I am adding edge(u,v) and edge (v,u) only one of them is
    # enumerated. 
    # If I only add (u,v), I can still query (u,v) and (v,u) using .get_edge_data()
    # self loops are also enumerated, hence I am eliminating them.
    G = nx.Graph()
    for i in range(number_of_riders):
        for j in range(number_of_riders):
                if j > i:
                    G.add_edge(str(i), str(j), weight=weights[(i, j)]) #why are node_ids strings TBD
    return G

def get_graph_weights(G,number_of_riders):
    #The following will give me n(n-1)/2 numbers. 
    #No numbers for self-loops since they are typically not present. TBD: assert no self loops
    weights0= nx.get_edge_attributes(G,'weight')

    #We will get n^2 numbers.
    MAX_WEIGHT = max([x[2]['weight'] for x in G.edges(data=True)])
    MIN_WEIGHT = min([x[2]['weight'] for x in G.edges(data=True)])
    OFFSET = 0
    if MIN_WEIGHT==0:
        OFFSET = MAX_WEIGHT/10
        MIN_WEIGHT = OFFSET
    DEFAULT_WEIGHT = MAX_WEIGHT * pow(number_of_riders, 3) #repeated in 
    weights = {}
    for i,j in weights0:
        weights[(i,j)] = weights0[(i,j)] + OFFSET
        weights[(j,i)] = weights0[(i,j)] + OFFSET
    for i in range(number_of_riders):
        weights[(i,i)] = DEFAULT_WEIGHT + OFFSET
    return MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights

def get_relabeled_graph(H,starting_index=0):
    label_mapping = {}
    nodes = H.nodes()
    for index in range(len(nodes)):
        label_mapping[nodes[index]] = int(index) + starting_index
    H = nx.relabel_nodes(H, label_mapping)
    return H
#---------------------- HELPER FUNCTIONS ---------------------------

#---------------------- RANDOM GRAPH ---------------------------

def generate_weights(number_of_riders):
    """
    Generating weights that will be used for the graph and ILP
    It generates same weight for both edges! 
    Both directional edges and self loops are weighted.
    """

    MAX_WEIGHT = 10
    MIN_WEIGHT = 1
    DEFAULT_WEIGHT = MAX_WEIGHT * pow(number_of_riders, 3)
    weights = {}
    for i in range(number_of_riders):
        for j in range(number_of_riders):
            if j < i:
                weights[(i, j)] = random.randint(MIN_WEIGHT, MAX_WEIGHT)
                weights[(j, i)] = weights[(i, j)]
            elif j == i:
                weights[(i, j)] = DEFAULT_WEIGHT

    #debug
    #pprint.pprint(weights)
    
    return MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights

def get_random_subgraph(G, number_of_nodes=100):
    nodes = G.nodes()
    subgraph_nodes = random.sample(nodes, number_of_nodes)
    H = G.subgraph(subgraph_nodes)
    H1 = get_relabeled_graph(H,starting_index=0)
    return H1
#---------------------- RANDOM GRAPH ---------------------------


def get_plots(result_stats=None, y_variable={'name': 'obj', 'label': 'Minimum Happiness',
                                             'title': 'Minimum happiness vs experiment number'}, 
                                             version="min",graphtype="random",plot_dir='.'):
    assert result_stats is not None

    if version=='avg':
        y_variable['title'] = 'Average happiness vs experiment number'
        y_variable['label'] = 'Average Happiness'


    unique_number_of_riders = result_stats.keys()
    min_number_of_riders = min(unique_number_of_riders)
    number_of_iterations = len([x for x in result_stats[min_number_of_riders]['greedy']])

    for number_of_riders in unique_number_of_riders:

        data = result_stats[number_of_riders]

        fig, ax = plt.subplots()
        index = np.arange(number_of_iterations)
        bar_width = 0.15
        opacity = 0.4
        color_scheme = ['b', 'r', 'g', 'y', 'k']
        offset = range(len(data))

        rects = {}
        for i, alg_type in enumerate(data.keys()):
            y_data = []
            for exp in data[alg_type]:
                y_data.append(exp[y_variable['name']])
            rects[alg_type] = plt.bar(index + offset[i] * bar_width,
                                      y_data,
                                      bar_width,
                                      alpha=opacity,
                                      color=color_scheme[i],
                                      label=alg_type)

        plt.xlabel('Exp #')
        plt.ylabel(y_variable['label'])
        plt.title(y_variable['title'])
        plt.xticks(index + bar_width, tuple(range(1, number_of_iterations + 1)))
        plt.legend()
        plt.tight_layout()
        # plt.show()

        fname = plot_dir+'/' + y_variable['name'] + '_' + str(number_of_riders) + '_' + version +'_'+ graphtype +'.png'
        print fname
        plt.savefig(fname)


def run_experiment(number_of_iterations=1, version="min",graphtype='random',rider_sizes = [pow(2, x) for x in range(5, 7)], GraphInput = None,temp_dir="."):
    group_size = 4
    result_stats = {}

    for number_of_riders in rider_sizes:
        print 'Number of riders: ', number_of_riders
        number_of_groups = number_of_riders / group_size  # should be a factor of N

        result_stats[number_of_riders] = {}
        result_stats[number_of_riders]['greedy'] = []
        result_stats[number_of_riders]['ILP'] = []

        for iteration in range(number_of_iterations):
            print 'Experiment: ', iteration

            if graphtype=='random':
                MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights = generate_weights(number_of_riders)
                G = get_rider_graph(number_of_riders, weights)
                print nx.info(G)
            elif graphtype=='facebook' or graphtype=='collab':
                #assume complete graph
                G = get_random_subgraph(GraphInput, number_of_riders)
                MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights = get_graph_weights(G,number_of_riders)
                
                # #debug
                # weightMat = np.zeros((number_of_riders,number_of_riders))
                # for k,v in weights.items():
                #     print k,":",v
                #     weightMat[k[0]][k[1]] = v
                # pprint.pprint(weightMat)

                G = get_rider_graph(number_of_riders, weights)
            else:
                return "Error: graphtype not found. getplot() will fail."


            result_greedy = solve_greedy(G, group_size, MAX_WEIGHT, version)
            result_ILP = solve_ILP(number_of_riders, number_of_groups, weights, MAX_WEIGHT, DEFAULT_WEIGHT, version,temp_dir)

            result_stats[number_of_riders]['greedy'].append(result_greedy)
            result_stats[number_of_riders]['ILP'].append(result_ILP)

    return result_stats


if __name__ == '__main__':
    temp_dir            = "group_formation_temp"
    plot_dir            = temp_dir  
    number_of_iterations= 10
    versions            = ['avg'] 
    graphtype           = 'collab'#'collab' #'random' #'facebook'
    rider_sizes         = [16,32] #[16,32]


    if graphtype=='random':
        print 'Random graphs will be generated on the fly.'
        GraphInput = None
    elif graphtype=='facebook':
        print "Loading facebook graph from pickle..."
        start_time_fb = time.time()
        GraphInput = nx.read_gpickle(temp_dir+"/fbgraph.gpickle")
        print "Loaded facebook in ", time.time() - start_time_fb
    elif graphtype=='collab':
        print "Loading modified collab graph from pickle..."
        start_time_collab = time.time()
        GraphInput = nx.read_gpickle(temp_dir+"/collabModified.gpickle")
        print "Loaded collabModified in ", time.time() - start_time_collab
    else:
        print "Error: graphtype not found. getplot() will fail."

    for version in versions:
        #CPLEX with TIMELIMIT needed!
        result_stats = run_experiment(number_of_iterations=number_of_iterations,
            version=version,
            graphtype=graphtype,
            rider_sizes=rider_sizes,
            GraphInput=GraphInput,
            temp_dir=temp_dir)
        get_plots(result_stats=result_stats,version=version,graphtype=graphtype,plot_dir=plot_dir)