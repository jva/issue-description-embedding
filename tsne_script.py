import json
import sys
from sklearn.manifold import TSNE
import numpy as np


# Receive the file path from command-line argument
file_path = sys.argv[1]

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert the list of lists to a numpy array
data = np.array(data)

# Apply t-SNE
tsne = TSNE(n_components=2)
transformed_data = tsne.fit_transform(data)

output = json.dumps(transformed_data.tolist())

print(output)
