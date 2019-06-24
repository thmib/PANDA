from __future__ import print_function, division
import tensorflow as tf
import networkx as nx
import pandas as pd
import numpy as np
import random
import sys
import os

from sklearn.model_selection import KFold
k_fold = KFold(5)


# DISCLAMER:
# Parts of this code are derived from
# https://github.com/williamleif/GraphSAGE
def zeros(shape, name=None):
    initials = tf.zeros(shape, dtype=tf.dtypes.float32)
    return tf.Variable(initials, name=name)


def glorot(shape, name=name):
    initial_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initials = tf.random_uniform(shape, minval=-initial_range, maxval=initial_range, dtype=tf.float32)
    return tf.Variable(initials, name=name)


def ones(shape, name=None):
    initials = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initials, name=name)


def uniform(shape, scale=0.05, name=None):
    initials = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initials, name=name)


def load_data(filename, norm=True):

    edges = pd.read_csv(filename)

    if os.path.exists(prefix + "-properties.npy"):
        properties = np.load(prefix + "-properties.npy")
    else:
        print("No features present.. Only identity features will be used.")
        properties = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not properties is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_properties = properties[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_properties)
        properties = scaler.transform(properties)

    return G, properties, id_map, class_map
