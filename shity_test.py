from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_list
from bmtk.utils.sim_setup import build_env_bionet
import numpy as np
from math import e
import pandas as pd

net = NetworkBuilder("biophysical")
numAAC = 1
numPyr = 1
numPV = 1

xside_length = 400; yside_length = 1000; height = 450; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(0, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

inds = np.random.choice(np.arange(0, np.size(pos_list, 0)), numAAC, replace=False)
pos = pos_list[inds, :]

# Place cell
net.add_nodes(N=numAAC, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)

pos_list = np.delete(pos_list, inds, 0)

inds = np.random.choice(np.arange(0, np.size(pos_list, 0)), numPyr, replace=False)
pos = pos_list[inds, :]

net.add_nodes(N=numPyr, pop_name='Pyr',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pyramidalcell',
              morphology=None)

pos_list = np.delete(pos_list, inds, 0)

inds = np.random.choice(np.arange(0, np.size(pos_list, 0)), numPyr, replace=False)
pos = pos_list[inds, :]

net.add_nodes(N=numPV, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)


def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=2):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    sid = src.node_id
    tid = trg.node_id

    src_pos = src['positions']
    trg_pos = trg['positions']
    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    print(dist)

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)

conn = net.add_edges(source={'pop_name': 'AAC'}, target={'pop_name': 'Pyr'},
                     iterator='one_to_one',
                     connection_rule=n_connections,
                     connection_params={'prob': 1},  # was.408
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params='AAC_To_PYR.json',
                     model_template='exp2syn',
                     distance_range=[0.0, 300.0],
                     target_sections=['axonal'],
                     sec_id=0,
                     sec_x=0.5)

conn = net.add_edges(source={'pop_name': 'Pyr'}, target={'pop_name': 'PV'},
                     iterator='one_to_one',
                     connection_rule=n_connections,
                     connection_params={'prob': 1},  # was.408
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params='AMPA_ExcToInh.json',
                     model_template='exp2syn',
                     distance_range=[0.0, 300.0],
                     target_sections=['somatic'],
                     sec_id=0,
                     sec_x=0.5)

net.build()
net.save(output_dir='network')

build_env_bionet(base_dir='./',
                network_dir='./network',
                tstop=1500.0, dt=0.1,
                report_vars=['v'],
                current_clamp={
                     'amp': 0.50,
                     'delay': 500.0,
                     'duration': 1000.0
                 },
                components_dir='biophys_components',
                config_file='simulation_config.json',
                compile_mechanisms=True)