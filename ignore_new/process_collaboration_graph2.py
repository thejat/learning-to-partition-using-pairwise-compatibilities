import networkx as nx
import itertools
import time

start_time_process = time.time()
temp_dir = '../../../group_formation_temp'

Collab = nx.read_gpickle(temp_dir+'/collab.gpickle')

MIN_WEIGHT = 10
MIN_NON_ZERO_WEIGHT = 10
for u,v,x in Collab.edges(data=True):
	MIN_WEIGHT = min(MIN_WEIGHT,x['value'])
	if x['value'] > 0:
		MIN_NON_ZERO_WEIGHT = min(MIN_NON_ZERO_WEIGHT,x['value'])

print "Found min weight ({0}) in the graph in absolute time: {1}".format(MIN_WEIGHT,time.time()-start_time_process)

if MIN_WEIGHT==0:
	MIN_WEIGHT = MIN_NON_ZERO_WEIGHT/2
print "MIN_WEIGHT modified to {0}".format(MIN_WEIGHT)

cliques = [x for x in nx.algorithms.find_cliques(Collab)]
cliques_filtered = [x for x in cliques if len(x) > 20]
node_list = list(set([node for clique in cliques_filtered for node in clique]))

print "Number of nodes remaining:",len(node_list)

CollabModified = nx.Graph()
i = 0
for u,v in itertools.combinations(node_list,2):
	i += 1
	if i%1000000==0:
		print "Iteration {0}".format(i)
	#need to change attribute from value to weight
	if Collab.has_edge(u,v)==True and Collab[u][v]['value'] > 0:
		CollabModified.add_edge(u,v,weight=Collab[u][v]['value'])
	else:
		CollabModified.add_edge(u,v,weight=MIN_WEIGHT*1.0/2)

print "Created CollabModified (absolute time: {0})".format(time.time()-start_time_process)

nx.write_gpickle(CollabModified,temp_dir+'/collabModified.gpickle')

print "Wrote CollabModified to disk in absolute time {0}".format(time.time()-start_time_process)
print "Info for original network:", nx.info(Collab)
print "\n----------------------------------------\n"
print "Info for the modified network:", nx.info(CollabModified)

MIN_WEIGHT = 10
MAX_WEIGHT = 0
for u,v,x in CollabModified.edges(data=True):
	MIN_WEIGHT = min(MIN_WEIGHT,x['weight'])
	MAX_WEIGHT = max(MAX_WEIGHT,x['weight'])

print "MIN_WEIGHT: {0}, MAX_WEIGHT: {1}".format(MIN_WEIGHT,MAX_WEIGHT)