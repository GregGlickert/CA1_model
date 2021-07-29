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
<<<<<<< Updated upstream
numAAC = 10
=======
numAAC = 147
>>>>>>> Stashed changes
numCCK = 10
numNGF = 10
numOLM = 10
numPV = 10
<<<<<<< Updated upstream
numPyr = 10
# % of cells in each layer
####SO####
AAC_in_SO = 0.238
CCK_in_SO = 0.217
OLM_in_SO = 1.0
PV_in_SO = 0.238
###SP###
AAC_in_SP = 0.70
CCK_in_SP = 0.261
PV_in_SP = 0.701
###SR###
AAC_in_SR = 0.062
CCK_in_SR = 0.325
NGF_in_SR = 0.17
PV_in_SR = 0.0596
###SLM###
CCK_in_SLM = 0.197
NGF_in_SLM = 0.83

# final cell amounts per layer
####SO####
AAC_in_SO = int(round(AAC_in_SO * numAAC))
CCK_in_SO = int(round(CCK_in_SO * numCCK))
OLM_in_SO = int(round(OLM_in_SO * numOLM))
PV_in_SO = int(round(PV_in_SO * numPV))
###SP###
AAC_in_SP = int(round(AAC_in_SP * numAAC))
CCK_in_SP = int(round(CCK_in_SP * numCCK))
PV_in_SP = int(round(PV_in_SP * numPV))
###SR###
AAC_in_SR = int(round(AAC_in_SR * numAAC))
CCK_in_SR = int(round(CCK_in_SR * numCCK))
NGF_in_SR = int(round(NGF_in_SR * numNGF))
PV_in_SR = int(round(PV_in_SR * numPV))
###SLM###
CCK_in_SLM = int(round(CCK_in_SLM * numCCK))
NGF_in_SLM = int(round(NGF_in_SLM * numNGF))


=======
numPyr = 31150
>>>>>>> Stashed changes
# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []
<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes

# Order from top to bottom is SO,SP,SR,SLM total
# SO layer
xside_length = 400; yside_length = 1000; height = 450; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(320, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SO = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SP layer
<<<<<<< Updated upstream
xside_length = 400; yside_length = 1000; height = 320; min_dist = 20
=======
xside_length = 400; yside_length = 1000; height = 320; min_dist = 8
>>>>>>> Stashed changes
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(290, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SP = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SR
xside_length = 400; yside_length = 1000; height = 290; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
<<<<<<< Updated upstream
z_grid = np.arange(79, height+min_dist, min_dist)
=======
z_grid = np.arange(80, height+min_dist, min_dist)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), AAC_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SO, pop_name='AAC',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numAAC_inSO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSO, pop_name='AAC',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(AAC_in_SO):
=======
for i in range(numAAC_inSO):
>>>>>>> Stashed changes
    cell_name.append("AAC in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SO, inds, 0)

# CCK basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), CCK_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SO, pop_name='CCK',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numCCK_inSO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSO, pop_name='CCK',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(CCK_in_SO):
=======
for i in range(numCCK_inSO):
>>>>>>> Stashed changes
    cell_name.append("CCK in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# OLM
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), OLM_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=OLM_in_SO, pop_name='OLM',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:olmcell',
              morphology=None)
# save location in array delete used locations
for i in range(OLM_in_SO):
    cell_name.append("OLM in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# PV
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), PV_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=PV_in_SO, pop_name='PV',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numPV_inSO, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=numPV_inSO, pop_name='PV',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(PV_in_SO):
=======
for i in range(numPV_inSO):
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), AAC_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SP, pop_name='AAC',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numAAC_inSP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSP, pop_name='AAC',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(AAC_in_SP):
=======
for i in range(numAAC_inSP):
>>>>>>> Stashed changes
    cell_name.append("AAC in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SP, inds, 0)

# CCK basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), CCK_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SP, pop_name='CCK',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numCCK_inSP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSP, pop_name='CCK',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(CCK_in_SP):
=======
for i in range(numCCK_inSP):
>>>>>>> Stashed changes
    cell_name.append("CCK in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# PV
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), PV_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# place cell
net.add_nodes(N=PV_in_SP, pop_name='PV',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numPV_inSP, replace=False)
pos = pos_list_SP[inds, :]

# place cell
net.add_nodes(N=numPV_inSP, pop_name='PV',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(PV_in_SP):
=======
for i in range(numPV_inSP):
>>>>>>> Stashed changes
    cell_name.append("PV in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# ############ SR LAYER ############ #
# AAC
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), AAC_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SR, pop_name='AAC',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numAAC_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numAAC_inSR, pop_name='AAC',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(AAC_in_SR):
=======
for i in range(numAAC_inSR):
>>>>>>> Stashed changes
    cell_name.append("AAC in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SR, inds, 0)

# CCK basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), CCK_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SR, pop_name='CCK',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numCCK_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSR, pop_name='CCK',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(CCK_in_SR):
=======
for i in range(numCCK_inSR):
>>>>>>> Stashed changes
    cell_name.append("CCK in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# NGF basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), NGF_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=NGF_in_SR, pop_name='NGF',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numNGF_inSR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numNGF_inSR, pop_name='NGF',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(NGF_in_SR):
=======
for i in range(numNGF_inSR):
>>>>>>> Stashed changes
    cell_name.append("NGF in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# PV
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), PV_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# place cell
net.add_nodes(N=PV_in_SR, pop_name='PV',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numPV_inSR, replace=False)
pos = pos_list_SR[inds, :]

# place cell
net.add_nodes(N=numPV_inSR, pop_name='PV',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(PV_in_SR):
=======
for i in range(numPV_inSR):
>>>>>>> Stashed changes
    cell_name.append("PV in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# ############ SLM LAYER ############ #

# CCK basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), CCK_in_SLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SLM, pop_name='CCK',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numCCK_inSLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numCCK_inSLM, pop_name='CCK',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(CCK_in_SLM):
=======
for i in range(numCCK_inSLM):
>>>>>>> Stashed changes
    cell_name.append("CCK in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)

# NGF basket
# Pick location
<<<<<<< Updated upstream
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), NGF_in_SLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=NGF_in_SLM, pop_name='NGF',
=======
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numNGF_inSLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numNGF_inSLM, pop_name='NGF',
>>>>>>> Stashed changes
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
<<<<<<< Updated upstream
for i in range(NGF_in_SLM):
=======
for i in range(numNGF_inSLM):
>>>>>>> Stashed changes
    cell_name.append("NGF in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)


df = pd.DataFrame(columns=('Cell type', 'X location', 'Y location', 'Z location'))
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

conn = net.add_edges(source={'pop_name': 'AAC'}, target={'pop_name': 'Pyr'},
                     connection_rule=n_connections,
                     connection_params={'prob': 0.05, 'max_dist': 500},  # was.408
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params='AAC_To_PYR.json',
                     model_template='exp2syn',
                     distance_range=[0.0, 500.0],
                     target_sections=['axonal'],
                     sec_id=0,
                     sec_x=0.5)

conn = net.add_edges(source={'pop_name': 'Pyr'}, target={'pop_name': 'AAC'},
                     connection_rule=n_connections,
                     connection_params={'prob': 0.007631, 'max_dist': 500},  # was.408
                     syn_weight=1,
                     delay=0.1,
                     dynamics_params='AMPA_ExcToInh.json',
                     model_template='exp2syn',
                     distance_range=[0.0, 500.0],
                     target_sections=['axonal'],
                     sec_id=0,
                     sec_x=0.5)

#connection_params={'a': 0.0382,'x0': 190,'sigma': 60,
#                                        'max_dist': 500},

def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=2):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)


##CONNECTIONS##
##AAC CONNECTIONS ###
net.add_edges(source=net.nodes(pop_name='AAC'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 1}, #was.408
              dynamics_params='GABA_InhToExc.json',
              model_template='Exp2Syn',
              syn_weight=1,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 3000.0])     # gets error saying cant find syn connection when axonal
###PV CONNECTIONS ###
net.add_edges(source=net.nodes(pop_name='PV'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 0.307},
              dynamics_params='GABA_InhToExc.json',
              model_template='Exp2Syn',
              syn_weight=1,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='PV'), target=net.nodes(pop_name='PV'),
              connection_rule=n_connections,
              connection_params={'prob': 0.706},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='PV'), target=net.nodes(pop_name='CCK'),
              connection_rule=n_connections,
              connection_params={'prob': 0.688},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='PV'), target=net.nodes(pop_name='AAC'),
              connection_rule=n_connections,
              connection_params={'prob': 0.705},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
###CCK CONNECTIONS###
net.add_edges(source=net.nodes(pop_name='CCK'), target=net.nodes(pop_name='OLM'),
              connection_rule=n_connections,
              connection_params={'prob': 0.366},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='CCK'), target=net.nodes(pop_name='PV'),
              connection_rule=n_connections,
              connection_params={'prob': 0.333},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='CCK'), target=net.nodes(pop_name='CCK'),
              connection_rule=n_connections,
              connection_params={'prob': 0.944},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='CCK'), target=net.nodes(pop_name='AAC'),
              connection_rule=n_connections,
              connection_params={'prob': 0.333},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='CCK'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 0.361},
              dynamics_params='GABA_InhToExc.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
###NGF CONNECTIONS###
net.add_edges(source=net.nodes(pop_name='NGF'), target=net.nodes(pop_name='NGF'),
              connection_rule=n_connections,
              connection_params={'prob': 0.475},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='NGF'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 0.391},
              dynamics_params='GABA_InhToExc.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
###OLM CONNECTIONS###
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='NGF'),
              connection_rule=n_connections,
              connection_params={'prob': 0.789},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='OLM'),
              connection_rule=n_connections,
              connection_params={'prob': 0.366},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='PV'),
              connection_rule=n_connections,
              connection_params={'prob': 0.487},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='CCK'),
              connection_rule=n_connections,
              connection_params={'prob': 0.2439}, #actually 2.439 but all r off rn
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='AAC'),
              connection_rule=n_connections,
              connection_params={'prob': 0.489},
              dynamics_params='GABA_InhToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='OLM'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 0.489},
              dynamics_params='GABA_InhToExc.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
###PYR CONNECTIONS###
net.add_edges(source=net.nodes(pop_name='Pyr'), target=net.nodes(pop_name='OLM'),
              connection_rule=n_connections,
              connection_params={'prob': 0.1624}, # actually 1.624
              dynamics_params='AMPA_ExcToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='Pyr'), target=net.nodes(pop_name='PV'),
              connection_rule=n_connections,
              connection_params={'prob': 0.136},
              dynamics_params='AMPA_ExcToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='Pyr'), target=net.nodes(pop_name='AAC'),
              connection_rule=n_connections,
              connection_params={'prob': 0.052},
              dynamics_params='AMPA_ExcToInh.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])
net.add_edges(source=net.nodes(pop_name='Pyr'), target=net.nodes(pop_name='Pyr'),
              connection_rule=n_connections,
              connection_params={'prob': 0.063},
              dynamics_params='AMPA_ExcToExc.json',
              model_template='Exp2Syn',
              syn_weight=6.0e-05,
              delay=2.0,
              target_sections=['somatic'],
              distance_range=[0.0, 300.0])



net.build()
net.save(output_dir='network')
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes

print(count)

build_env_bionet(base_dir='./',
                network_dir='./network',
<<<<<<< Updated upstream
                tstop=1500.0, dt=0.1,
                report_vars=['v'],
                components_dir='biophys_components',
                config_file='simulation_config.json',
                compile_mechanisms=True)
#                 current_clamp={
#                     'amp': 0.800,
#                     'delay': 500.0,
#                     'duration': 1000
#                 },

=======
                config_file='config.json',
                tstop=2000.0, dt=0.1,
                report_vars=['v'],
                components_dir='biophys_components',
                current_clamp={
                     'amp': 0.500,
                     'delay': 500.0,
                     'duration': 1000.0,
                     'gids': [0, 10, 11, 15, 31313, 31314]
                 },
                compile_mechanisms=False)
>>>>>>> Stashed changes



