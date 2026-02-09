import os
from dotenv import load_dotenv
from openai import Client
import numpy as np
import uuid
import faiss
import pickle

load_dotenv()

# def build_index():
api_key = os.getenv("OPENAI_API_KEY")

client = Client(api_key=api_key)

with open("knowledge.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Im chunking it per paragraphs 
chunks = [
    chunk.strip()
    for chunk in content.split("\n\n")
    if chunk.strip()
]

vectors = []
ids = []
metadata = {}

batch_size = 100

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]

    response = client.embeddings.create(
        input=batch,
        model="text-embedding-3-small"
    )

    for chunk_text, item in zip(batch, response.data):
        embedding = np.array(item.embedding, dtype="float32")
        vector_id = uuid.uuid4().int & (1<<63)-1

        vectors.append(embedding)
        ids.append(vector_id)
        metadata[vector_id] = chunk_text
    
vectors = np.vstack(vectors)
ids = np.array(ids, dtype="float32")

dimension = vectors.reshape[1]
faiss.normalize_L2(dimension)

base_index = faiss.IndexFlatIP(dimension)
index = faiss.IndexIDMap(base_index)
index.add_with_ids(vectors, ids)

faiss.write_index(index, "data/faiss.index")

with open("data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)



