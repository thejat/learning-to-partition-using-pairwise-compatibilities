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
    H = G.copy() #This is the graph that stores previous meta round weights
    for u,v in H.edges():
        H[u][v]['weight'] = 0

    commuters = [i for i in range(graph_size)]

    #new
    observed_groups = dict()
    number_of_sample_rounds = 100
    #new ends

    for round_number in range(number_of_rounds):
        #Shuffle the commuters and break them into random groups
        permuted_commuter_list = copy.deepcopy(commuters)
        random.shuffle(permuted_commuter_list)
        groups_unfiltered = break_list_into_groups(permuted_commuter_list, group_size)


        #filtering 

        groups = []
        for group in groups_unfiltered:
            #only pick those groups that have not been observed before
            if tuple(group) not in observed_groups:
                groups.append(group)
                observed_groups[tuple(group)] = 1 #mark as observed 

        # groups = groups_unfiltered



        #true and noisy feedback

        true_feedback = {}
        for group in groups: #for each of the group
            for u in group: #for every rider in the group
                true_feedback[u] = MAX_WEIGHT
                for v in group:
                    if u != v:
                        true_feedback[u] = min(true_feedback[u],G[u][v]['weight'])

        #get feedback from all users averaged over a new number of rounds
        
        noisy_feedback_all = {}
        for group in groups: #for each of the group
            for u in group: #for every rider in the group
                noisy_feedback_all[u] = []
        for sample_round in range(number_of_sample_rounds):
            for group in groups: #for each of the group
                for u in group: #for every rider in the group
                    noisy_feedback_all[u].append(true_feedback[u] + 1.0*random.uniform(-1.0*MAX_WEIGHT/10,1.0*MAX_WEIGHT/10))
        noisy_feedback = {}
        for group in groups: #for each of the group
            for u in group: #for every rider in the group
                noisy_feedback[u] = 1.0*sum(noisy_feedback_all[u])/number_of_sample_rounds
                #assert abs(noisy_feedback[u]-true_feedback[u]) < 1e-3

        # for u in G.nodes():
        #     if u in true_feedback:
        #         noisy_feedback[u] = true_feedback[u]

        # groups
        for group in groups: #for each of the group
            for u in group: #for every rider in the group
                for v in group:
                    if u != v:
                        H[u][v]['weight'] = max(H[u][v]['weight'], noisy_feedback[u])

        weights_per_round[round_number] = [x['weight'] for u,v,x in H.edges(data=True)]

        #print "Number of observed groups is:",len(observed_groups)

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


def l2_error_normalized(estimate, original, observable):
    return numpy.linalg.norm((numpy.asarray(estimate)-numpy.asarray(original))*numpy.asarray(observable))*1.0/numpy.linalg.norm(numpy.asarray(original)*numpy.asarray(observable))

#----------------------- explore only: learn weights general ends ------------------

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
        performanceL2 = [0] * number_of_rounds
        for round_number in range(number_of_rounds):
            performance[round_number] = number_of_matching_weights(weights_per_round[round_number, :], original_weight_array, observable_weights,tolerance=0.01)
            
            performanceL2[round_number] = l2_error_normalized(weights_per_round[round_number, :], original_weight_array, observable_weights)

        #Display output
        no_observable_weights = sum(observable_weights)
        no_estimated_weights = performance[-1]

        return {'total_weights':len(observable_weights),
        'no_observable_weights':no_observable_weights,
        'no_estimated_weights':no_estimated_weights,
        'no_estimated_weights_midway' : performance[number_of_rounds/2 - 1],
        'success': 1.0 * no_estimated_weights / no_observable_weights,
        'performance_match': performance,
        'performance_errors': performanceL2}
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





def get_plot_weights_general_error(performance, rounds, graphtype='random', graph_size=8, plot_dir='.'):

    #Plot of number of weights learned as a function of round number
    #Example Usage: get_plot_weights_general_error(performance, range(1, number_of_rounds + 1), graphtype, len(G), plot_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = rounds
    ys = performance
    ax.plot(xs, ys)
    plt.show()

    plt.xlabel('Round #')
    plt.ylabel('Error')
    plt.title('Error between estimates and ground truth over rounds')
    plt.tight_layout()
    plt.ylim((0,1.2*max(ys)))

    fname = plot_dir + '/learning_weights_' + str(graphtype) + '_' + str(graph_size) + '.png'
    print "plot will be saved at:",fname
    plt.savefig(fname)



def get_plot_weights_general_nips(result_log):


    """
    Creating grouped stacked bar plot
    """
    # setting up data
    performanceL2_all = dict()

    graphtypes = result_log.keys()
    for graphtype in graphtypes:
        performanceL2_all[graphtype] = dict()
        for number_of_riders in sorted(result_log[graphtype].keys()):

            fig = plt.figure()
            ax = fig.add_subplot(111)

            performanceL2_all[graphtype][number_of_riders] = []
            for run_number in sorted(result_log[graphtype][number_of_riders].keys()):
                performanceL2_all[graphtype][number_of_riders].append(result_log[graphtype][number_of_riders][run_number]['performance_errors'])

            temp = numpy.asarray(performanceL2_all[graphtype][number_of_riders])
            #print temp

            temp_avg = numpy.mean(temp, axis=0)
            #print temp_avg
            temp_std = numpy.std(temp, axis=0)

            xs = range(len(temp_avg))
            ys = temp_avg

            ax.fill_between(xs, ys-temp_std, ys+temp_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            ax.plot(xs, ys)



        plt.show()
        plt.xlabel('Round #')
        plt.ylabel('Error')
        plt.title('Error between estimates and ground truth over rounds')
        plt.tight_layout()
        plt.ylim((0,1.2*max(ys)))

        plot_dir = '.'
        fname = plot_dir + '/learning_weights_' + str(graphtype) + '_' + str(number_of_riders) + '.png'
        print "plot will be saved at:",fname
        plt.savefig(fname)



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

    graphtypes          = ['random', 'facebook'] #"random"#"facebook" #'collab'
    no_MC               =  10
    group_size          =  4
    rider_sizes         = [32]
    number_of_rounds    = 400 # 6*nCr(rider_sizes[0], group_size) / (rider_sizes[0]/ group_size)

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
                        H,MAX_WEIGHT,weights = get_instance(graphtype,FaceBook,Collab,number_of_riders)
                        result_log = wrapper_evaluation(H,
                            group_size,
                            graphtype=graphtype,
                            MAX_WEIGHT=MAX_WEIGHT,
                            evaluation_function=evaluation_function,
                            number_of_rounds=number_of_rounds)
                        all_result_log[graphtype][number_of_riders][run_number] = copy.deepcopy(result_log)
        
        pickle.dump(all_result_log, open(pickle_file, 'wb'))

    #get_plots(all_result_log)
    # get_plot_weights_general_error(all_result_log['random'][16][0]['performance_errors'],range(1, number_of_rounds + 1) , graphtype='random', graph_size=16, plot_dir='.')
    # # get_plot_weights_general_error(all_result_log['random'][16][0]['performance_match'],range(1, number_of_rounds + 1) , graphtype='random2', graph_size=16, plot_dir='.')
    get_plot_weights_general_nips(all_result_log)

    print "Experiment finished. End time: {0}".format(time.time()-start_time_exp)

