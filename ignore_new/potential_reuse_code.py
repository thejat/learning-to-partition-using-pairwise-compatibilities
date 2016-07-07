#----------------------- reuse -----------------------------------
def noise_model(name,MAX_WEIGHT):
    if name=="gaussian":
        return random.gauss(0,MAX_WEIGHT/10)
    elif name=="uniform":
        return random.uniform(-1.0*MAX_WEIGHT/10,1.0*MAX_WEIGHT/10)
    elif name=="none":
        return 0
    else:
        print "Error in noise model. Misspecified. Returns 0."
        return 0

def get_plot_weights_general(performance, rounds, graphtype='random', graph_size=8, plot_dir='.'):

    #Plot of number of weights learned as a function of round number
    #Example Usage: get_plot_weights_general(performance, range(1, number_of_rounds + 1), graphtype, len(G), plot_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = rounds
    ys = performance
    ax.plot(xs, ys)
    plt.show()

    plt.xlabel('Round #')
    plt.ylabel('Number of weights learned')
    plt.title('Number of weights learned over rounds')
    plt.tight_layout()
    plt.ylim((0,1.2*max(ys)))

    fname = plot_dir + '/learning_weights_' + str(graphtype) + '_' + str(graph_size) + '.png'
    plt.savefig(fname)
#----------------------- reuse ends ------------------------------