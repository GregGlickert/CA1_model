from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot
from bmtk.analyzer.spike_trains import to_dataframe
import pandas
import matplotlib.pyplot as plt

_ = plot_raster(config_file='CA1_config.json',group_by='pop_name',title="raster", show=False)

#plot_rates_boxplot(config_file='CA1_config.json', group_by='pop_name', title='boxplot', show=False)

_ = plot_traces(config_file='CA1_config.json', node_ids=[8], report_name='v_report', title='voltage report for PN',
                show=False)

_ = plot_traces(config_file='CA1_config.json', node_ids=[0], report_name='v_report', title='voltage report for AAC',
                show=False)

_ = plot_traces(config_file='CA1_config.json', group_by='pop_name', report_name='v_report',
                show=False)

#df = to_dataframe(config_file='config.json')

#df.sort_values(by=['node_ids'])

#df.to_csv('spikedata.csv')

#print(len(df))
plt.show()