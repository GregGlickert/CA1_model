import os, sys
from bmtk.simulator import bionet
import numpy as np
import synapses
import warnings
from bmtk.simulator.bionet.pyfunction_cache import add_weight_function

def run(config_file):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    synapses.load()

    def lognormal(edge_props, source, target):
        m = edge_props["syn_weight"]
        s = edge_props["weight_sigma"]
        mean = np.log(m) - 0.5 * np.log((s/m)**2+1)
        std = np.sqrt(np.log((s/m)**2 + 1))
        return np.random.lognormal(mean, std, 1)

    add_weight_function(lognormal)

    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()
    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)

    sim.run()
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('CA1_config.json')