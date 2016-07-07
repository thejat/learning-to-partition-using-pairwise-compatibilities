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

CollabModified = nx.Graph()
i = 0
node_list = Collab.nodes()[:5000]
for u,v in itertools.combinations(node_list,2):
	i += 1
	if i%1000000==0:
		print "Iteration {0}".format(i)
	#need to change attribute from value to weight
	if Collab.has_edge(u,v)==True and Collab[u][v]['value'] > 0:
		CollabModified.add_edge(u,v,weight=Collab[u][v]['value'])
	else:
		CollabModified.add_edge(u,v,weight=MIN_WEIGHT*1.0/10)

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

'''

Collaboration graph: http://networkdata.ics.uci.edu/data/cond-mat-2005/

IMPORTANT NOTE: 0th step is not shown above where the gml file was read, the following changes were made and a gpickle file was written.
Manual Changes:
 - H[17][37693]['value'] = 0
 - H[1771][1772]['value'] = 0

The output of the above run:

In [2]: run -i process_collaboration_graph.py
Found min weight (0) in the graph in absolute time: 2.5119998455
MIN_WEIGHT modified to 0.0172414
Iteration 1000000
Iteration 2000000
Iteration 3000000
Iteration 4000000
Iteration 5000000
Iteration 6000000
Iteration 7000000
Iteration 8000000
Iteration 9000000
Iteration 10000000
Iteration 11000000
Iteration 12000000
Created CollabModified (absolute time: 51.7149999142)
Wrote CollabModified to disk in absolute time 180.041999817
Info for original network: Name:
Type: Graph
Number of nodes: 40421
Number of edges: 175693
Average degree:   8.6932

----------------------------------------

Info for the modified network: Name:
Type: Graph
Number of nodes: 5000
Number of edges: 12497500
Average degree: 4999.0000
MIN_WEIGHT: 0.00172414, MAX_WEIGHT: 46


'''