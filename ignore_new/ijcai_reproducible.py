import sys, collections, math, itertools, numpy, matplotlib, copy, pprint, json, random, time, pickle
from group_formation_experiments import get_rider_graph, generate_weights, nCr, get_relabeled_graph, get_random_subgraph,get_graph_weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import seaborn as sns

matplotlib.use('Agg')
random.seed(111)

__author__ = 'q4BKGR0Q,q4fj4lj9'

#----------------------- explore only: learn weights general -----------------------
def break_list_into_groups(permuted_commuter_list, group_size):
    return [permuted_commuter_list[i:i + group_size] for i in range(0, len(permuted_commuter_list), group_size)]

def learn_weights_general(G, group_size,MAX_WEIGHT,number_of_rounds=1):
    graph_size = len(G)
    number_of_node_pairs = len(G.edges())

    weights_per_round = numpy.zeros(shape=(number_of_rounds, number_of_node_pairs))

    #initialize
    H = G.copy()
    for u,v in H.edges():
        H[u][v]['weight'] = 0

    commuters = [i for i in range(graph_size)]

    for round_number in range(number_of_rounds):
        #Shuffle the commuters and break them into random groups
        permuted_commuter_list = copy.deepcopy(commuters)
        random.shuffle(permuted_commuter_list)
        groups = break_list_into_groups(permuted_commuter_list, group_size)

        # groups
        for group in groups: #for each of the group
            for u in group: #for every rider in the group
                #his feedback is the minimum of his group neighborhood
                feedback = MAX_WEIGHT
                for v in group:
                    if u != v:
                        feedback = min(feedback,G[u][v]['weight'])
                #propagate this feedback to our estimate of weights
                for v in group:
                    if u !=v:
                        H[u][v]['weight'] = max(H[u][v]['weight'], feedback)


        weights_per_round[round_number] = [x['weight'] for u,v,x in H.edges(data=True)]

    original_weight_array = [x['weight'] for u,v,x in G.edges(data=True)]
    return [weights_per_round, original_weight_array, number_of_rounds]

def get_observable_weights_new(G, group_size,MAX_WEIGHT):

    #initialize
    H = G.copy()
    for u,v in H.edges():
        H[u][v]['observable'] = 0

    #For every group, figure out the edges that are observed
    for group in itertools.combinations(G.nodes(), group_size):
        for u in group: #for every rider in the group
            #her/his feedback is the minimum of her/his group neighborhood
            min_weight_by_user = MAX_WEIGHT
            for v in group:
                if u != v:
                    min_weight_by_user = min(min_weight_by_user,G[u][v]['weight'])
            #propagate this feedback to our estimate of weights
            for v in group:
                if u !=v and G[u][v]['weight'] == min_weight_by_user:
                    H[u][v]['observable'] = 1

    observable_indicator_array = [x['observable'] for u,v,x in H.edges(data=True)]
    #print observable_indicator_array, len(observable_indicator_array)
    return observable_indicator_array

def number_of_matching_weights(estimate, original, observable, tolerance = 0.01):
    match = 0
    for i, e in enumerate(estimate):
        if observable[i] == 1:
            if numpy.absolute(e-original[i]) <= tolerance:
                match += 1
    return match


#----------------------- explore only: learn weights general ends ------------------


def learn_weights_scores(G, group_size,MAX_WEIGHT,number_of_rounds=1):
    return None


def wrapper_evaluation(H, group_size=4,graphtype="random",MAX_WEIGHT=10000,evaluation_function=learn_weights_general,number_of_rounds=1):

    #Relabeling the nodes!
    G = get_relabeled_graph(H,starting_index=0)
    
    #The algorithm: Learn the weights or explore-exploit
    [weights_per_round, original_weight_array, number_of_rounds] = evaluation_function(G, group_size, MAX_WEIGHT,
        number_of_rounds=number_of_rounds)

    if evaluation_function.__name__=="learn_weights_general":

        #For performance purposes: assess which weights are observable out of the total set of weights
        observable_weights = get_observable_weights_new(G,group_size,MAX_WEIGHT) #0-1 valued array
        performance = [0] * number_of_rounds
        for round_number in range(number_of_rounds):
            performance[round_number] = number_of_matching_weights(weights_per_round[round_number, :], original_weight_array, observable_weights,tolerance=0.01)


        #Display output
        no_observable_weights = sum(observable_weights)
        no_estimated_weights = performance[-1]
        # print "Max observable weights are {0} out of {1}. Alg estimates {2} of them by round {3}. Success {4}" \
        #     .format(no_observable_weights, len(observable_weights), no_estimated_weights, number_of_rounds,
        #             1.0 * no_estimated_weights / no_observable_weights)

        return {'total_weights':len(observable_weights),
        'no_observable_weights':no_observable_weights,
        'no_estimated_weights':no_estimated_weights,
        'no_estimated_weights_midway' : performance[number_of_rounds/2 - 1],
        'success': 1.0 * no_estimated_weights / no_observable_weights}
    else:
        return None

def load_graph(graphtype=None,temp_dir='.'):
    if graphtype=='facebook':
        #Load original graph
        print "Loading facebook graph from pickle..."
        start_time_fb = time.time()
        GraphInput = nx.read_gpickle(temp_dir+"/fbgraph.gpickle")
        print "Facebook graph loaded in sec:",time.time() - start_time_fb
    elif graphtype=='collab':
        print "Loading modified collab graph from pickle..."
        start_time_collab = time.time()
        GraphInput = nx.read_gpickle(temp_dir+"/collabModified.gpickle")
        # GraphInput = nx.read_gpickle(temp_dir+"/collab.gpickle")
        print "Loaded collabModified in ", time.time() - start_time_collab
    elif graphtype=="random":
        print "Random graph will be generated on the fly."
        GraphInput = None
    else:
        print "Error: Incorrect graphtype."
        GraphInput = None

    return GraphInput


def get_instance(graphtype,FaceBook,Collab,number_of_riders):
    if graphtype=="random":
        #Get a random weighted graph realization H
        MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights = generate_weights(number_of_riders)
        H = get_rider_graph(number_of_riders, weights)
    elif graphtype=="facebook" or graphtype=="collab":
        if graphtype=='facebook':
            H = get_random_subgraph(FaceBook, number_of_riders)
        else:
            H = get_random_subgraph(Collab,number_of_riders)

        MAX_WEIGHT, MIN_WEIGHT, DEFAULT_WEIGHT, weights = get_graph_weights(H,number_of_riders)
        H = get_rider_graph(number_of_riders, weights)
    else:
        print "graphtype is misspecified."
        H =None
        MAX_WEIGHT = None
    return H,MAX_WEIGHT,weights


def get_plots(result_log):
    """
    Creating grouped stacked bar plot
    """
    # setting up data
    total_observable_weights = dict()
    no_estimated_weights = dict()
    no_estimated_weights_midway = dict()

    graphtypes = result_log.keys()
    for graphtype in graphtypes:
        total_observable_weights[graphtype] = []
        no_estimated_weights[graphtype] = []
        no_estimated_weights_midway[graphtype] = []

        for number_of_riders in sorted(result_log[graphtype].keys()):
            total_observable_weights[graphtype].append(result_log[graphtype][number_of_riders][0]['no_observable_weights'])
            no_estimated_weights[graphtype].append(result_log[graphtype][number_of_riders][0]['no_estimated_weights'])
            no_estimated_weights_midway[graphtype].append(result_log[graphtype][number_of_riders][0]['no_estimated_weights_midway'])
    

    # percentage conversion
    for graphtype in graphtypes:
        no_estimated_weights_midway[graphtype] = 100.0 * numpy.asarray(no_estimated_weights_midway[graphtype]) / numpy.asarray(total_observable_weights[graphtype])
        no_estimated_weights[graphtype] = 100.0 * numpy.asarray(no_estimated_weights[graphtype]) / numpy.asarray(total_observable_weights[graphtype])
        total_observable_weights[graphtype] = [100.0] * len(total_observable_weights[graphtype])

    # print total_observable_weights, no_estimated_weights, no_estimated_weights_midway

    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context("talk")

        # setting up plot parameters
        xlabels     = sorted(result_log[graphtypes[0]].keys())
        bar_width   = 0.35
        # epsilon     = 0.015
        epsilon     = 0.015
        line_width  = 1
        opacity     = 0.7
        random_bar_positions = 2 * numpy.arange(1, len(total_observable_weights['random'])+1)
        collab_bar_positions = random_bar_positions + bar_width
        fb_bar_positions = collab_bar_positions + bar_width

        color = {}
        color['random'] = '#ff0000'
        color['collab'] = '#6ACC65'
        color['fb']     = '#1e90ff'
        
        for number_of_riders in result_log[graphtypes[0]].keys():
            # random
            # bar for estimated halfway
            random_estimated_halfway_bar = plt.bar(random_bar_positions, 
                no_estimated_weights_midway['random'],
                bar_width,
                color=color['random'],
                hatch="//")

            # bar for estimated
            random_estimated_bar = plt.bar(random_bar_positions,
                numpy.asarray(no_estimated_weights['random']) - numpy.asarray(no_estimated_weights_midway['random']),
                bar_width-epsilon,
                bottom=no_estimated_weights_midway['random'],
                alpha=opacity,
                color="white",
                edgecolor=color['random'],
                linewidth=line_width,
                hatch="//")

            # bar for total observable
            random_total_observable_bar = plt.bar(random_bar_positions,
                numpy.asarray(total_observable_weights['random']) - numpy.asarray(no_estimated_weights['random']),
                bar_width-epsilon,
                bottom=no_estimated_weights['random'],
                alpha=opacity,
                color="white",
                edgecolor=color['random'],
                linewidth=line_width,
                hatch="0")

            # collab
            # bar for estimated halfway
            collab_estimated_halfway_bar = plt.bar(collab_bar_positions, 
                no_estimated_weights_midway['collab'],
                bar_width,
                color=color['collab'],
                hatch="//")

            # bar for estimated
            collab_estimated_bar = plt.bar(collab_bar_positions,
                numpy.asarray(no_estimated_weights['collab']) - numpy.asarray(no_estimated_weights_midway['collab']),
                bar_width-epsilon,
                bottom=no_estimated_weights_midway['collab'],
                alpha=opacity,
                color="white",
                edgecolor=color['collab'],
                linewidth=line_width,
                hatch="//")

            # bar for total observable
            collab_total_observable_bar = plt.bar(collab_bar_positions,
                numpy.asarray(total_observable_weights['collab']) - numpy.asarray(no_estimated_weights['collab']),
                bar_width-epsilon,
                bottom=no_estimated_weights['collab'],
                alpha=opacity,
                color="white",
                edgecolor=color['collab'],
                linewidth=line_width,
                hatch="0")

            # # facebook
            fb_estimated_halfway_bar = plt.bar(fb_bar_positions, 
                no_estimated_weights_midway['facebook'],
                bar_width,
                color=color['fb'],
                hatch="//")

            # bar for estimated
            fb_estimated_bar = plt.bar(fb_bar_positions,
                numpy.asarray(no_estimated_weights['facebook']) - numpy.asarray(no_estimated_weights_midway['facebook']),
                bar_width-epsilon,
                bottom=no_estimated_weights_midway['facebook'],
                alpha=opacity,
                color="white",
                edgecolor=color['fb'],
                linewidth=line_width,
                hatch="//")

            # bar for total observable
            fb_total_observable_bar = plt.bar(fb_bar_positions,
                numpy.asarray(total_observable_weights['facebook']) - numpy.asarray(no_estimated_weights['facebook']),
                bar_width-epsilon,
                bottom=no_estimated_weights['facebook'],
                alpha=opacity,
                color="white",
                edgecolor=color['fb'],
                linewidth=line_width,
                hatch="0")            
        
        plt.xticks(collab_bar_positions + bar_width/2, xlabels)
        plt.yticks(numpy.arange(0, 101, 10))
        plt.xlabel('Number of riders')
        plt.ylabel('Weights')
        plt.ylim(0, 105)

        # legends
        colored = patches.Patch(color='gray', label='Weights estimated by T/2 rounds')
        hatched = patches.Rectangle([20, 20], width=0.25, height=0.1, facecolor='white', edgecolor='gray', hatch='//', label='Weights estimated by T rounds')
        blank = patches.Rectangle([20, 20], width=0.25, height=0.1, facecolor='white', edgecolor='gray', label='Total observable weights')
        random = patches.Patch(color=color['random'], label='Randomly generated graph')
        collab = patches.Patch(color=color['collab'], label='Collaborative network')
        fb = patches.Patch(color=color['fb'], label='Facebook social graph')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125),
          ncol=3, fancybox=True, shadow=True, handles = [colored, random, hatched, collab, blank, fb])
        
        plt.grid(linestyle='--')
        plt.savefig("learning-weights.png")


if __name__ == '__main__':
    #Settings
    if sys.argv[1] == 'learn_weights_general':
        evaluation_function = learn_weights_general
        pickle_file = 'result_log_learn_weights_general.pickle'
        print "Setting: Learn General Weights"
    elif sys.argv[1] == 'learn_weights_scores':
        evaluation_function = learn_weights_scores
        pickle_file = 'result_log_learn_weights_scores.pickle'
        print "Setting: Learn Scores"
    else:
        print "Provide arg: learn_weights_general or learn_weights_scores. Exiting."
        exit(1)

    start_time_exp      = time.time()

    graphtypes          = ['random', 'collab', 'facebook'] #"random"#"facebook" #'collab'
    no_MC               =  1
    group_size          =  4
    rider_sizes         = [16,32,64,128]
    number_of_rounds    = 1000 # 6*nCr(rider_sizes[0], group_size) / (rider_sizes[0]/ group_size)

    #Load all datasets
    FaceBook = None
    Collab = None
    if 'facebook' in graphtypes:
        FaceBook = load_graph('facebook',"group_formation_temp")
    if 'collab' in graphtypes:
        Collab = load_graph('collab',"group_formation_temp")

    all_result_log = {}
    for graphtype in graphtypes:
        print "graph type is ",graphtype
        all_result_log[graphtype] = {}
        for number_of_riders in rider_sizes:
                print "number of riders is ", number_of_riders
                all_result_log[graphtype][number_of_riders] = {}
                for run_number in range(no_MC):
                        print "monte carlo run number: {0}. Time: {1}".format(run_number,time.time()-start_time_exp)
                        H,MAX_WEIGHT = get_instance(graphtype,FaceBook,Collab,number_of_riders)
                        result_log = wrapper_evaluation(H,
                            group_size,
                            graphtype=graphtype,
                            MAX_WEIGHT=MAX_WEIGHT,
                            evaluation_function=evaluation_function,
                            number_of_rounds=number_of_rounds)
                        all_result_log[graphtype][number_of_riders][run_number] = copy.deepcopy(result_log)
        
        pickle.dump(all_result_log, open(pickle_file, 'wb'))

    get_plots(all_result_log)
    print "Experiment finished. End time: {0}".format(time.time()-start_time_exp)