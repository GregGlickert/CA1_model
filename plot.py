from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot
from bmtk.analyzer.spike_trains import to_dataframe
import pandas as pd
import matplotlib.pyplot as plt
import h5py

#raster = plot_raster(config_file='simulation_config.json', group_by='pop_name', times=(150, 300), title="raster", show=False)

#plt.savefig('raster.png')

def raster(spikes_df, node_set, skip_ms=0, ax=None):
    spikes_df = spikes_df[spikes_df['timestamps'] > skip_ms]
    for node in node_set:
        cells = range(node['start'], node['end'] + 1)  # +1 to be inclusive of last cell
        cell_spikes = spikes_df[spikes_df['node_ids'].isin(cells)]

        ax.scatter(cell_spikes['timestamps'], cell_spikes['node_ids'],
                   c='tab:' + node['color'], s=4, label=node['name'])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    ax.grid(False)

dt = 0.1
steps_per_ms = 1/dt
skip_seconds = 5
skip_ms = skip_seconds*1000

spikes_location = 'output/spikes.h5'

print("loading " + spikes_location)
f = h5py.File(spikes_location)
spikes_df = pd.DataFrame({'node_ids':f['spikes']['biophysical']['node_ids'],'timestamps':f['spikes']['biophysical']['timestamps']})
print("done")

node_set = [
    {"name": "PN", "start": 331, "end": 31480, "color": "blue"},
    {"name": "PV", "start": 199, "end": 330, "color": "red"},
    {"name": "PV", "start": 31584, "end": 31971, "color": "red"},
    {"name": "PV", "start": 31981, "end": 32013, "color": "red"},
    {"name": "SOM", "start": 35, "end": 198, "color": "green"},
    {"name": "AAC", "start": 0, "end": 34, "color": "olive"},
    {"name": "AAC", "start": 31481, "end": 31583, "color": "olive"},
    {"name": "AAC", "start": 31972, "end": 31980, "color": "olive"}
]

fig, ax1 = plt.subplots(1,1,figsize=(12,4.8))

raster(spikes_df,node_set,skip_ms=0,ax=ax1)

plt.savefig('raster.png')
