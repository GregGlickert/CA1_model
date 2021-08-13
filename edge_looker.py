import h5py
import glob
import pprint
import numpy as np
import pandas as pd

from sonata.circuit import File


# TO MAKE SURE WEIGHTS ARE GETTING SAVED CORRECT
path = "updated_conns/biophysical_biophysical_edges.h5"

f = h5py.File(path)

print(f['edges']['biophysical_biophysical'].keys())

print(f['edges']['biophysical_biophysical']['0']['syn_weight'][:])

#print(f['edges']['biophysical_to_biophysical'].keys())

#print(f['edges']['biophysical_to_biophysical']['target_node_id'][:])


# FOR SEEING WHAT IS CONNECTED TO WHAT
"""
net = File(data_files=['network/biophysical_biophysical_edges.h5', 'network/biophysical_nodes.h5'],
           data_type_files=['network/biophysical_biophysical_edge_types.csv', 'network/biophysical_node_types.csv'])

print('Contains nodes: {}'.format(net.has_nodes))
print('Contains edges: {}'.format(net.has_edges))


file_edges = net.edges
print('Edge populations in file: {}'.format(file_edges.population_names))
recurrent_edges = file_edges['biophysical_to_biophysical']

con_count = 0
for edge in recurrent_edges.get_target(15):  # we can also use get_targets([id0, id1, ...])
    assert (edge.target_node_id == 15)
    con_count += 1

print('There are {} connections onto target node #{}'.format(con_count, 15))
"""
