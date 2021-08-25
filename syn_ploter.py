import h5py
import matplotlib.pyplot as plt

path = "updated_conns/biophysical_biophysical_edges.h5"
f = h5py.File(path, 'r')

weights = f['edges']['biophysical_biophysical']['0']['syn_weight'][:]

source_id = f['edges']['biophysical_biophysical']['target_node_id'][:]

start_of_pyr = 35
end_of_pyr = 31184

pyr_weights = []
aac_weights = []
for i in range(len(weights)):
    if source_id[i] > 35 and source_id[i] < 31184:
        pyr_weights.append(weights[i])
    else:
        aac_weights.append(weights[i])
plt.hist(pyr_weights)
plt.title('syn_weight for synapses targeting PNS')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()

plt.hist(aac_weights)
plt.title('syn_weight for synapses targeting AAC')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()
