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


#pour télécharger l'analyse fonctionel du dataset
#cleaner.download_functional_data()

"""# Download nucleus data (if not already downloaded)
cleaner.download_nucleus_data()


cleaner.download_functional_data()

funcreader = mic.MicronsFunctionalReader(datadir="data")

# Extract neurons and segments
units, segments = funcreader.process_nucleus_data()
units.head()"""


funcreader = mic.MicronsFunctionalReader("data")
hash = funcreader.get_hashes_by_session("4_7")[0]

response = funcreader.get_responses_by_hash(hash, brain_area='V1')



print(response[0])

print(response[0]['responses'][5, :])

print(len(response[0]['responses'][5, :]))

import matplotlib.pyplot as plt
import numpy as np

x = response[0]['responses'][5, :]
# Tracer l'histogramme
plt.figure(figsize=(8,5))
plt.plot(x)
plt.show()
