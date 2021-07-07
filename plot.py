from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
from bmtk.analyzer.spike_trains import plot_raster
_ = plot_traces(config_file='simulation_config.json', report_name='v_report', node_ids=[0])
#_ = plot_rates(config_file='simulation_config.json')
#_ = plot_raster(config_file='simulation_config.json',group_by='pop_name')

_ = plot_traces(config_file='simulation_config.json', report_name='v_report',
                node_ids=[16,17,18,19,20,21,22,23,24,25])

