from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_list
from bmtk.utils.sim_setup import build_env_bionet
import numpy as np
import pandas as pd

net = NetworkBuilder("biophysical")
# amount of cells
numAAC = 10
numCCK = 10
numNGF = 10
numOLM = 10
numPV = 10
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


# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []

# Order from top to bottom is SO,SP,SR,SLM total
# SO layer
xside_length = 400; yside_length = 1000; height = 450; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(320, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SO = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SP layer
xside_length = 400; yside_length = 1000; height = 320; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(290, height+min_dist, min_dist)
xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
pos_list_SP = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# SR
xside_length = 400; yside_length = 1000; height = 290; min_dist = 20
x_grid = np.arange(0, xside_length+min_dist, min_dist)
y_grid = np.arange(0, yside_length+min_dist, min_dist)
z_grid = np.arange(79, height+min_dist, min_dist)
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
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), AAC_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SO, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(AAC_in_SO):
    cell_name.append("AAC in SO layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SO, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), CCK_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SO, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(CCK_in_SO):
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
inds = np.random.choice(np.arange(0, np.size(pos_list_SO, 0)), PV_in_SO, replace=False)
pos = pos_list_SO[inds, :]

# place cell
net.add_nodes(N=PV_in_SO, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(PV_in_SO):
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
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), AAC_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SP, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(AAC_in_SP):
    cell_name.append("AAC in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SP, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), CCK_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SP, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(CCK_in_SP):
    cell_name.append("CCK in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SP, 0)), PV_in_SP, replace=False)
pos = pos_list_SP[inds, :]

# place cell
net.add_nodes(N=PV_in_SP, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(PV_in_SP):
    cell_name.append("PV in SP layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SP, inds, 0)

# ############ SR LAYER ############ #
# AAC
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), AAC_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=AAC_in_SR, pop_name='AAC',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)
# save location in array delete used locations
for i in range(AAC_in_SR):
    cell_name.append("AAC in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list = np.delete(pos_list_SR, inds, 0)

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), CCK_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SR, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(CCK_in_SR):
    cell_name.append("CCK in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), NGF_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# Place cell
net.add_nodes(N=NGF_in_SR, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(NGF_in_SR):
    cell_name.append("NGF in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# PV
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SR, 0)), PV_in_SR, replace=False)
pos = pos_list_SR[inds, :]

# place cell
net.add_nodes(N=PV_in_SR, pop_name='PV',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pvbasketcell',
              morphology=None)
# save location in array delete used locations
for i in range(PV_in_SR):
    cell_name.append("PV in SR layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SR, inds, 0)

# ############ SLM LAYER ############ #

# CCK basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), CCK_in_SLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=CCK_in_SLM, pop_name='CCK',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:cckcell',
              morphology=None)
# save location in array delete used locations
for i in range(CCK_in_SLM):
    cell_name.append("CCK in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)

# NGF basket
# Pick location
inds = np.random.choice(np.arange(0, np.size(pos_list_SLM, 0)), NGF_in_SLM, replace=False)
pos = pos_list_SLM[inds, :]

# Place cell
net.add_nodes(N=NGF_in_SLM, pop_name='NGF',
              positions=positions_list(positions=pos),
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:ngfcell',
              morphology=None)
# save location in array delete used locations
for i in range(NGF_in_SLM):
    cell_name.append("NGF in SLM layer")
    cell_x.append(pos[i][0])
    cell_y.append(pos[i][1])
    cell_z.append(pos[i][2])

pos_list_SO = np.delete(pos_list_SLM, inds, 0)


df = pd.DataFrame(columns=('Cell type', 'X location', 'Y location', 'Z location'))
for i in range(len(cell_name)):
    df.loc[i] = [cell_name[i], cell_x[i], cell_y[i], cell_z[i]]
df.to_csv("cell_locations.csv")

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



build_env_bionet(base_dir='./',
                network_dir='./network',
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




