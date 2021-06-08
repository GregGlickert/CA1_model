from bmtk.builder import NetworkBuilder
import numpy as np
import sys
import synapses


net = NetworkBuilder("biophysical")

#PV BASKET
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:pvbasketcell',
#        morphology=None)

#CHN
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:axoaxoniccell',
#        morphology=None)

#CCK basket
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:cckcell',
#        morphology=None)

#OLM
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:olmcell',
#        morphology=None)

#NGF
#net.add_nodes(
#        mem_potential='e',
#        model_type='biophysical',
#        model_template='hoc:ngfcell',
#        morphology=None)

#PYR
net.add_nodes(
        mem_potential='e',
        model_type='biophysical',
        model_template='hoc:pyramidalcell',
        morphology=None)


net.build()
net.save_nodes(output_dir='network')
from bmtk.utils.sim_setup import build_env_bionet
build_env_bionet(base_dir='./',
                network_dir='./network',
                tstop=1000.0, dt=0.1,
                report_vars=['v'],
                current_clamp={
                    'amp':.300,
                    'delay': 100,
                    'duration':600
                },
                components_dir='biophys_components',
                compile_mechanisms=True)




