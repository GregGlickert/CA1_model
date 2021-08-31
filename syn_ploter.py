import h5py
import matplotlib.pyplot as plt
import seaborn as sns

path = "updated_conns/biophysical_biophysical_edges.h5"
f = h5py.File(path, 'r')

weights = f['edges']['biophysical_biophysical']['0']['syn_weight'][:]

source_id = f['edges']['biophysical_biophysical']['target_node_id'][:]

start_of_pyr = 35
end_of_pyr = 1035

pyr_weights = []
aac_weights = []
for i in range(len(weights)):
    if source_id[i] > start_of_pyr and source_id[i] < end_of_pyr:
        pyr_weights.append(weights[i])
    else:
        aac_weights.append(weights[i])
plt.hist(pyr_weights)
plt.title('syn_weight from bmtk targeting PNS')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()

plt.hist(aac_weights)
plt.title('syn_weight from bmtk targeting AAC')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()

path = "output/syns_chn2pyr.h5"
f = h5py.File(path, 'r')

W = (f['report']['biophysical']['data'][:])
weight = []
print(len(W[0]))
for i in range(len(W[0])):
    weight.append(W[0][i])

sns.set_style('darkgrid')
sns.histplot(weight, kde=True)
plt.title('syn_weight from inside mod file targeting PNS')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()


path = 'output/syns_pyr2int_ampa.h5'
f = h5py.File(path, 'r')

W = (f['report']['biophysical']['data'][:])
weight = []
for i in range(len(W[0])):
    weight.append(W[0][i])

sns.set_style('darkgrid')
sns.histplot(weight, kde=True)
plt.title('syn_weight from inside mod file W_ampa targeting AAC')
plt.ylabel('# of synapses')
plt.xlabel('syn_weight')
plt.show()


