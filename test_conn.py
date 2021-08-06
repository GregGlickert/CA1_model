from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_list
from bmtk.utils.sim_setup import build_env_bionet
from math import exp
import numpy as np
import pandas as pd
import random

seed = 999
random.seed(seed)
np.random.seed(seed)

net = NetworkBuilder("biophysical")
# amount of cells
numAAC = 10  # 147
numCCK = 10  # 360
numNGF = 10  # 580
numOLM = 10  # 164
numPV = 10  # 553
numPyr = 31150  # 31150
# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []
# amount of cells per layer
numAAC_inSO = int(round(numAAC*0.238))
numAAC_inSP = int(round(numAAC*0.7))
numAAC_inSR = int(round(numAAC*0.062))
numCCK_inSO = int(round(numCCK*0.217))
numCCK_inSP = int(round(numCCK*0.261))
numCCK_inSR = int(round(numCCK*0.325))
numCCK_inSLM = int(round(numCCK*0.197))
numNGF_inSR = int(round(numNGF*0.17))
numNGF_inSLM = int(round(numNGF*0.83))
numPV_inSO = int(round(numPV*0.238))
numPV_inSP = int(round(numPV*0.701))
numPV_inSR = int(round(numPV*0.0596))
print("placing cells")
# total 400x1000x450
# Order from top to bottom is SO,SP,SR,SLM total
# SO layer
xside_length = 400; yside_length = 1000; height = 450; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(320, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SO = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SP layer
xside_length = 400; yside_length = 1000; height = 320; min_dist = 8
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(290, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SP = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SR
xside_length = 400; yside_length = 1000; height = 290; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(80, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SR = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SLM
xside_length = 400; yside_length = 1000; height = 79; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(0, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SLM = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# ############ SO LAYER ############ #
# AAC
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numAAC_inSO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSO, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC_inSO):
    cell_name.append("AAC in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SO, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numCCK_inSO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSO, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK_inSO):
    cell_name.append("CCK in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# OLM
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numOLM, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=numOLM, pop_name='OLM',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:olmcell',
              morphology=None)
# save location in array delete used locations
for i in range(numOLM):
    cell_name.append("OLM in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numPV_inSO, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=numPV_inSO, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV_inSO):
    cell_name.append("PV in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# ############ SP LAYER ############ #
# PV
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numPyr, replace=False)
pos = pos_list_SP[inds, :]

net.add_nodes(N=numPyr, pop_name='Pyr',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pyramidalcell',
              morphology=None)
for i in range(numPyr):
    cell_name.append("Pyr in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SP, inds, 0)
# AAC
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numAAC_inSP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSP, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC_inSP):
    cell_name.append("AAC in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SP, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numCCK_inSP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSP, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK_inSP):
    cell_name.append("CCK in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numPV_inSP, replace=False)
pos = pos_list_SP[inds, :]

# place cell
net.add_nodes(N=numPV_inSP, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV_inSP):
    cell_name.append("PV in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# ############ SR LAYER ############ #
# AAC
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numAAC_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSR, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC_inSR):
    cell_name.append("AAC in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SR, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numCCK_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSR, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK_inSR):
    cell_name.append("CCK in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numNGF_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numNGF_inSR, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(numNGF_inSR):
    cell_name.append("NGF in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numPV_inSR, replace=False)
pos = pos_list_SR[inds, :]

# place cell
net.add_nodes(N=numPV_inSR, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV_inSR):
    cell_name.append("PV in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# ############ SLM LAYER ############ #

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numCCK_inSLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSLM, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK_inSLM):
    cell_name.append("CCK in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numNGF_inSLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numNGF_inSLM, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(numNGF_inSLM):
    cell_name.append("NGF in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)


df = pd.DataFrame(columns=('Cell type','X location', 'Y location', 'Z location'))
for i in range(len(cell_name)):
    df.loc[i] = [cell_name[i], cell_x[i], cell_y[i], cell_z[i]]
df.to_csv("cell_locations.csv")
count = 0
def AAC_to_PYR(src, trg, a, x0, sigma, max_dist):
    if src.node_id == trg.node_id:
        return 0

    sid = src.node_id
    tid = trg.node_id

    src_pos = src['positions']
    trg_pos = trg['positions']

    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    prob = a * exp(-((dist - x0) ** 2) / (2 * sigma ** 2))
    #print(dist)
    #prob = (prob/100)
    #print(prob)

    if dist <= max_dist:
        global count
        count = count + 1
    if dist <= max_dist and np.random.uniform() < prob:
        connection = 1
        #print("creating {} synapse(s) between cell {} and {}".format(1,sid,tid))
    else:
        connection = 0
    return connection

def density(src, trg, max_dist, a, sigma, x0):
    if src.node_id == trg.node_id:
        return 0

    sid = src.node_id
    tid = trg.node_id

    src_pos = src['positions']
    trg_pos = trg['positions']

    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    prob = a * exp(-((dist - x0) ** 2) / (2 * sigma ** 2))
    # print(dist)
    # prob = (prob/100)
    # print(prob)
    if dist <= max_dist:
        global count
        count = count + 1
    if dist <= max_dist and np.random.uniform() < prob:
        connection = 1
        # print("creating {} synapse(s) between cell {} and {}".format(1,sid,tid))
    else:
        connection = 0
    return connection


def n_connections(src, trg, max_dist, prob=0.1):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    src_pos = src['positions']
    trg_pos = trg['positions']
    dist = np.sqrt((src_pos[0] - trg_pos[0]) ** 2 + (src_pos[1] - trg_pos[1]) ** 2 + (src_pos[2] - trg_pos[2]) ** 2)
    if dist <= max_dist:
        if np.random.uniform() > prob:
            return 0
        else:
            return 1

#conn = net.add_edges(source={'pop_name': 'AAC'}, target={'pop_name': 'Pyr'},
#                     connection_rule=n_connections,
#                     connection_params={'prob': 0.05, 'max_dist': 500},  # was.408
#                     syn_weight=1,
#                     delay=0.1,
#                     dynamics_params='AAC_To_PYR.json',
#                     model_template='exp2syn',
#                     distance_range=[0.0, 500.0],
#                     target_sections=['axonal'],
#                     sec_id=0,
#                     sec_x=0.5)

#conn = net.add_edges(source={'pop_name': 'Pyr'}, target={'pop_name': 'AAC'},
#                     connection_rule=n_connections,
#                     connection_params={'prob': 0.007631, 'max_dist': 500},
#                     syn_weight=1,
#                     delay=0.1,
#                     dynamics_params='AMPA_ExcToInh.json',
#                     model_template='exp2syn',
#                     distance_range=[0.0, 500.0],
#                     target_sections=['apical'],
#                     sec_id=0,
#                     sec_x=0.5)
print("looking at density")
conn = net.add_edges(source={'pop_name': 'Pyr'}, target={'pop_name': 'Pyr'},
                     connection_rule=density,
                     connection_params={'a': 0.0382,'x0': 190,'sigma': 60,
                                                             'max_dist': 500},
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params='AMPA_ExcToExc.json',
                     model_template='exp2syn',
                     distance_range=[0.0, 500.0],
                     target_sections=['apical'],
                     sec_id=0,
                     sec_x=0.5)



net.build()
net.save(output_dir='network')

print(count)

build_env_bionet(base_dir='./',
                network_dir='./network',
                config_file='config.json',
                tstop=2000.0, dt=0.1,
                report_vars=['v'],
                components_dir='biophys_components',
                current_clamp={
                     'amp': 0.500,
                     'delay': 500.0,
                     'duration': 100.0,
                     'gids': [0, 1, 2]
                 },
                compile_mechanisms=False)



