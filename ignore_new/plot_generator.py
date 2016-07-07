import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

plot_dir = 'group_formation_temp'

def autolabel(axes, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if (rect.get_y() < 0):
        	axes.text(rect.get_x() + rect.get_width()/2., -1.35*height,
	                '%d' % int(height),
	                ha='right', va='bottom')
        else:
	        axes.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%d' % int(height),
	                ha='center', va='bottom')

def get_plots(result_stats=None, graphtypes=None,version='avg'):
	# graphtypes = result_stats.keys()
	unique_number_of_riders = result_stats[graphtypes[0]].keys()
	min_number_of_riders = min(unique_number_of_riders)
	number_of_iterations = len([x for x in result_stats[graphtypes[0]][min_number_of_riders]['greedy']])

	percentage_diff = np.zeros(shape=(len(graphtypes), len(unique_number_of_riders)))

	means = dict()
	std = dict()

	for graphtype in graphtypes:
		means[graphtype] = np.zeros(shape=len(unique_number_of_riders))
		std[graphtype] = np.zeros(shape=len(unique_number_of_riders))

		k = 0
		for i in unique_number_of_riders:
			percentage_diff = np.zeros(shape=number_of_iterations)
			for j in range(number_of_iterations):
				percentage_diff[j] = 100.0 * (result_stats[graphtype][i]['ILP'][j]['obj'] - result_stats[graphtype][i]['greedy'][j]['obj']) / result_stats[graphtype][i]['ILP'][j]['obj']
			means[graphtype][k] = percentage_diff.mean(0)
			std[graphtype][k] = percentage_diff.std(0)
			
			k += 1


	# print means, std
	# print mins, maxes, means, std

	with sns.axes_style("white"):
		sns.set_style("ticks")
		sns.set_context("talk")

		# barplot with standard deviation
		# setting up plot parameters
		labels 		= unique_number_of_riders
		fig, axes 	= plt.subplots()
		ind 		= np.arange(len(unique_number_of_riders))
		width 		= 0.1

		colors 		= ['#ff0000', '#6ACC65', '#1e90ff'] # sequence: random, collab, fb

		k = 0
		bp = [None] * len(graphtypes)
		for graphtype in graphtypes:
			bp[k] = axes.bar(ind + width * k, means[graphtype], width, color=colors[k], yerr=std[graphtype])
			autolabel(axes, bp[k])
			k += 1

		axes.set_ylabel('Percentage difference')
		axes.set_xlabel('Number of riders')
		# axes.set_title('Percentage difference (ILP, Greedy) vs. Number of riders')
		axes.set_xticks(ind + width * 1.5)
		axes.set_xticklabels(unique_number_of_riders)
		axes.legend(bp, graphtypes, loc='upper center', bbox_to_anchor=(0.5, 1.105), ncol=3)
		plt.ylim(-100, 55)
		plt.grid()
		plt.savefig('percentage-difference.png')


    # fname = plot_dir+'/' + y_variable['name'] + '_' + str(number_of_riders) + '_' + version +'_'+ graphtype + '_boxplot_' +'.png'
    # print fname
    # plt.savefig(fname)

if __name__ == '__main__':
	versions 	= ['avg']
	graphtypes 	= ['random', 'collab', 'facebook']

	result_stats = dict()
	for graphtype in graphtypes:
		if graphtype=='random':
			result_stats[graphtype] = pickle.load(open('results_random.pickle', 'rb'))
		elif graphtype=='facebook':
			result_stats[graphtype] = pickle.load(open('results_facebook.pickle', 'rb'))
		elif graphtype=='collab':
			result_stats[graphtype] = pickle.load(open('results_collab.pickle', 'rb'))
		else:
			print "Error: graphtype not found. getplot() will fail."

	for version in versions:
		get_plots(result_stats=result_stats, graphtypes=graphtypes, version=version)