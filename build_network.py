from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_list
from bmtk.utils.sim_setup import build_env_bionet
import numpy as np
import pandas as pd

net = NetworkBuilder("biophysical")
# amount of cells
numAAC = 1
numCCK = 1
numNGF = 1
numOLM = 1
numPV = 1
numPyr = 1
# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []

# Order from top to bottom is SO,SP,SR,SLM
# SO layer
xside_length = 1400; yside_length = 1400; height = 400; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(300, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SO = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SP layer
xside_length = 1400; yside_length = 1400; height = 300; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(200, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SP = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SR
xside_length = 1400; yside_length = 1400; height = 200; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(100, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SR = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SLM
xside_length = 1400; yside_length = 1400; height = 100; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(0, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SLM = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# ############ SO LAYER ############ #
# AAC
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numAAC, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numAAC, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC):
    cell_name.append("AAC in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SO, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numCCK, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=numCCK, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK):
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
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), numPV, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=numPV, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV):
    cell_name.append("PV in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SO, inds, 0)

# ############ SP LAYER ############ #
# PV
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numPyr, replace=False)
pos = pos_list_SP[inds, :]

net.add_nodes(N=numPV, pop_name='Pyr',
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
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numAAC, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numAAC, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC):
    cell_name.append("AAC in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SP, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numCCK, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=numCCK, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK):
    cell_name.append("CCK in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), numPV, replace=False)
pos = pos_list_SP[inds, :]

# place cell
net.add_nodes(N=numPV, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV):
    cell_name.append("PV in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# ############ SR LAYER ############ #
# AAC
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numAAC, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numAAC, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC):
    cell_name.append("AAC in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SR, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numCCK, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numCCK, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK):
    cell_name.append("CCK in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numNGF, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=numNGF, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC):
    cell_name.append("NGF in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), numPV, replace=False)
pos = pos_list_SR[inds, :]

# place cell
net.add_nodes(N=numPV, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(numPV):
    cell_name.append("PV in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# ############ SLM LAYER ############ #

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numCCK, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numCCK, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(numCCK):
    cell_name.append("CCK in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), numNGF, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=numNGF, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(numAAC):
    cell_name.append("NGF in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)


df = pd.DataFrame(columns=('Cell type','X location', 'Y location', 'Z location'))
for i in range(len(cell_name)):
    df.loc[i] = [cell_name[i], cell_x[i], cell_y[i], cell_z[i]]
df.to_csv("cell_locations.csv")

net.build()
net.save_nodes(output_dir='network')



build_env_bionet(base_dir='./',
                network_dir='./network',
                tstop=500.0, dt=0.1,
                report_vars=['v'],
                components_dir='biophys_components',
                compile_mechanisms=True)


