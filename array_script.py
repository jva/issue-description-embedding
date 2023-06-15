from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import json
import sys

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def calculate_embedding(text):
    # Perform the embedding calculation for the text string
    embedding = embeddings.embed_query(text)
    return embedding

# Receive the file path from command-line argument
file_path = sys.argv[1]

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Calculate embeddings for each text string
embeddings = [calculate_embedding(text) for text in data]

# Return the embeddings as an array of arrays
output = json.dumps(embeddings)
print(output)



