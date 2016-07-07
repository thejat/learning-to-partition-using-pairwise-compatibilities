
import os
import time

import networkx as nx
import numpy

from learning_weights import explore_weights

__author__ = 'asmita'


def get_similarity(l1, l2):
    a = 0
    b = 0
    c = 0
    for e1, e2 in itertools.izip(l1, l2):
        if e1 == 1 and e2 == 1:
            a += 1
        elif e1 == 0 and e2 == 1:
            b += 1
        elif e1 == 1 and e2 == 0:
            c += 1
    return get_jaccard(a, b, c)


def get_jaccard(a, b, c):
    if a == 0 and b == 0 and c == 0:
        return 0
    return 10.0 * a / (a + b + c)



EDGE_FILE = "data/fb/facebook_combined.txt"
FEATURE_DIRECTORY = "data/fb/facebook"
COMPLETE_FEATURE_DIRECTORY = "data/fb/complete_features"

G = nx.Graph()
feature_dict = {}
feature_dict_reverse = {}
ego_feature_dict = {}
ego_feature_dict_reverse = {}
number_of_vertices = 0
feature_matrix = []


def load_data():
    load_egos()
    load_vertices()
    global feature_matrix
    feature_matrix = numpy.zeros(shape=(number_of_vertices, len(feature_dict)))
    load_all_features()


def load_egos():
    feature_name_position = 0
    global ego_feature_dict, ego_feature_dict_reverse, feature_dict, feature_dict_reverse
    for f in os.listdir(FEATURE_DIRECTORY):
        if f.endswith(".featnames"):
            ego = int(os.path.splitext(f)[0])
            with open(os.path.join(FEATURE_DIRECTORY, f)) as feature_name_file:
                for line in feature_name_file:
                    if line.strip():
                        split = line.split(' ', 1)
                        position = int(split[0])
                        feature_name = split[1].strip()
                        if ego not in ego_feature_dict:
                            ego_feature_dict[ego] = {}
                        ego_feature_dict[ego][position] = feature_name
                        if ego not in ego_feature_dict_reverse:
                            ego_feature_dict_reverse[ego] = {}
                        ego_feature_dict_reverse[ego][feature_name] = position
                        if feature_name not in feature_dict:
                            feature_dict[feature_name] = feature_name_position
                            feature_dict_reverse[feature_name_position] = feature_name
                            feature_name_position += 1


def load_vertices():
    global number_of_vertices
    vertices = []
    for f in os.listdir(FEATURE_DIRECTORY):
        if f.endswith(".feat"):
            u = os.path.splitext(f)[0]
            vertices.append(u)
            with open(os.path.join(FEATURE_DIRECTORY, f)) as feature_file:
                for line in feature_file:
                    if line.strip():
                        v = line.split(' ', 1)[0]
                        vertices.append(v)
    number_of_vertices = len(set(vertices))


def load_all_features():
    global feature_matrix, ego_feature_dict, feature_dict
    for f in os.listdir(FEATURE_DIRECTORY):
        if f.endswith(".feat"):
            ego = int(os.path.splitext(f)[0])
            with open(os.path.join(FEATURE_DIRECTORY, f)) as feature_name_file:
                for line in feature_name_file:
                    if line.strip():
                        split = line.split()
                        alter = int(split[0])
                        feature_list = [int(i) for i in split[1:]]
                        for i in range(len(feature_list)):
                            feature_name = ego_feature_dict[ego][i]
                            feature_matrix_index = feature_dict[feature_name]
                            feature_matrix[alter][feature_matrix_index] = feature_list[i]


def build_graph():
    global G, feature_matrix

    print 'Generating complete graph...'
    start_time = time.time()
    G = nx.complete_graph(number_of_vertices)
    print 'Time elapsed {0} seconds'.format(time.time() - start_time)

    print 'Computing weights'
    start_time = time.time()
    for u, v, d in G.edges(data=True):
        d['weight'] = get_similarity(feature_matrix[u], feature_matrix[v]) * 10
    print 'Time elapsed {0} seconds'.format(time.time() - start_time)

    print 'Writing out graph...'
    start_time = time.time()
    nx.write_gpickle(G, "../../../group_formation_temp/fbgraph.gpickle")
    print 'Time elapsed {0} seconds'.format(time.time() - start_time)


if __name__ == "__main__":

    print 'Loading data...'
    start_time = time.time()
    load_data()
    print 'Time elapsed {0} seconds'.format(time.time() - start_time)
    print 'Building graph...'
    start_time = time.time()
    build_graph()
    print 'Time elapsed {0} seconds'.format(time.time() - start_time)

    # G = nx.read_gpickle("fbgraph.gpickle")

    print 'Graph size: ',  len(G)
    print 'Edges: ', len(G.edges())
    #
    # start_time = time.time()
    # explore_weights(G)
    # print 'Time elapsed {0} seconds'.format(time.time() - start_time)