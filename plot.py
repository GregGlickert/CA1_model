from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot
from bmtk.analyzer.spike_trains import to_dataframe
import pandas

_ = plot_raster(config_file='config.json',group_by='pop_name')

plot_rates_boxplot(config_file='config.json', group_by='pop_name')

_ = plot_traces(config_file='config.json', node_ids=[126], report_name='v_report')

_ = plot_traces(config_file='config.json', node_ids=[0], report_name='v_report')

df = to_dataframe(config_file='config.json')

df.sort_values(by=['node_ids'])

df.to_csv('spikedata.csv')

print(len(df))
