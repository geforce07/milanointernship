import microns_datacleaner as mic
import microns_datacleaner.filters as fl
from caveclient import CAVEclient
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# Initialize CAVE client (authentication token can be saved if needed)
client = CAVEclient(server_address="https://global.daf-apis.com")
# client.auth.save_token(token="f00ee338f3e5c5a1b75da5013ab2badc")

# Specify data folder and dataset version
cleaner = mic.MicronsDataCleaner(datadir="data", version=1300, download_policy='minimum')

# Download nucleus data (if not already downloaded)
cleaner.download_nucleus_data()


cleaner.download_functional_data()

funcreader = mic.MicronsFunctionalReader(datadir="data")






# Extract neurons and segments
units, segments = cleaner.process_nucleus_data(functional_data='match')

# Filter excitatory neurons in V1 layer 2/3 with proofread axons
v1_l23_neurons = fl.filter_neurons(
    units,
    tuning='matched',
    layer=['L2/3'],
    brain_area=['V1'],
    proofread='ax_clean'
)

# Keep only excitatory neurons
v1_l23_neurons = v1_l23_neurons[v1_l23_neurons['classification_system'] == 'excitatory_neuron']

print(f"Number of L2/3 excitatory neurons with proofread: {len(v1_l23_neurons)}")

x = v1_l23_neurons['pt_position_x'].values
y = v1_l23_neurons['pt_position_y'].values
z = v1_l23_neurons['pt_position_z'].values
coords = np.column_stack([x, y, z])

# Compute all pairwise distances
dist_matrix = squareform(pdist(coords, metric='euclidean'))
print("Distance matrix ready:", dist_matrix.shape)



pre_ids = v1_l23_neurons['pt_root_id']
cleaner.download_synapse_data(pre_ids, pre_ids)
cleaner.merge_synapses(syn_table_name="example_merged_synapses")
synapses = pd.read_csv("data/1300/raw/example_merged_synapses.csv")
print(synapses)

v1_synapses = fl.synapses_by_id(
    synapses,
    pre_ids=pre_ids,
    post_ids=pre_ids,
    both=True   # only keep synapses where pre AND post are in our list
)


pt_root_ids = v1_l23_neurons['pt_root_id'].tolist()
#Construction of the connectivity Matrix
C = np.zeros((len(pt_root_ids), len(pt_root_ids)), dtype=np.int8)
for i, pre_id in enumerate(pt_root_ids):
    for j, post_id in enumerate(pt_root_ids):
        # Check if there is a synapse from pre_id to post_id
        found = v1_synapses[
            (v1_synapses['pre_pt_root_id'] == pre_id) &
            (v1_synapses['post_pt_root_id'] == post_id)
        ]
        if len(found) > 0:
            C[i, j] = 1
np.fill_diagonal(C, 0) #Faut-il garder les autapse ?
print("Connectivity matrix ready:", C.shape)

import matplotlib.pyplot as plt
import numpy as np

# Mask distances selon connectivité
connected_distances = dist_matrix[C == 1]
unconnected_distances = dist_matrix[C == 0]

# Définir des bins pour l'histogramme
bins = np.linspace(0, dist_matrix.max(), 100)

# Tracer l'histogramme
plt.figure(figsize=(8,5))
plt.hist(connected_distances, bins=bins, alpha=0.6, label='Connected', histtype='step') #density true = 
#plt.hist(unconnected_distances, bins=bins, alpha=0.3, label='Not Connected')
plt.xlabel("Distance between neurons (mm)")
plt.ylabel("Number of neuron pairs")
plt.title("Distribution of distances vs connectivity")
plt.legend()
plt.show()



