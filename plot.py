from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_rates
_ = plot_traces(config_file='simulation_config.json', node_ids=[0], report_name='v_report')
_ = plot_rates(config_file='simulation_config.json')