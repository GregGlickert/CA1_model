from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot
from bmtk.analyzer.spike_trains import to_dataframe
import pandas
import matplotlib.pyplot as plt
import h5py

_ = plot_raster(config_file='CA1_config.json', group_by='pop_name', title="raster")

plot_rates_boxplot(config_file='CA1_config.json', group_by='pop_name', title='boxplot')

_ = plot_traces(config_file='CA1_config.json', node_ids=[925], report_name='v_report', title='voltage report for PN')

_ = plot_traces(config_file='CA1_config.json', node_ids=[15], report_name='v_report', title='voltage report for AAC')

#_ = plot_traces(config_file='CA1_config.json', group_by='pop_name', report_name='v_report')

#df = to_dataframe(config_file='config.json')

#df.sort_values(by=['node_ids'])

#df.to_csv('spikedata.csv')

#print(len(df))