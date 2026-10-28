import microns_datacleaner as mic
import microns_datacleaner.filters as fl
import microns_datacleaner.remapper as rem
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

# 1. Télécharger les données fonctionnelles (vous l'avez déjà fait)
funcreader = mic.MicronsFunctionalReader(datadir="data")

# 2. Traiter les données nucléaires en incluant les données fonctionnelles
#    C'est l'étape clé ! Le paramètre 'functional_data' doit être 'match'
units, segments = cleaner.process_nucleus_data(functional_data='all')

# 3. Filtrer les neurones comme vous le faites déjà
v1_l23_neurons = fl.filter_neurons(
    units,
    tuning='matched',
    layer=['L2/3'],
    brain_area=['V1'],
    proofread='ax_clean'
)
v1_l23_neurons = v1_l23_neurons[v1_l23_neurons['classification_system'] == 'excitatory_neuron']

# 4. Maintenant, accédez à rate_ori !
#    Après l'étape 2, la colonne 'rate_ori' devrait être présente dans 'units'
#    et donc dans 'v1_l23_neurons' si elle existe pour les neurones sélectionnés.

print(v1_l23_neurons.columns.tolist())  # Affiche les premières valeurs

# ... (suite de votre code pour la connectivité)