import os
from dotenv import load_dotenv
from openai import Client
import numpy as np
import uuid
import faiss
import pickle
import pandas as pd
from pypdf import PdfReader
import sys


def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

load_dotenv()

# def build_index():
api_key = os.getenv("OPENAI_API_KEY")

client = Client(api_key=api_key)

reader = PdfReader("kb/Cybersecurity (Amendment) Draft Bill 2025 final 15102025.pdf")

texts = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        texts += page_text + "\n"




# Im chunking it per paragraphs 
# chunks = [
#     chunk.strip()
#     for chunk in texts.split("\n\n")
#     if chunk.strip()
# ]

chunks = chunk_text(texts)

# print(chunks)
# sys.exit()

vectors = []
ids = []
metadata = {}

batch_size = 10

# for i in range(0, len(chunks), batch_size):
#     batch = chunks[i:i+batch_size]

for chunk in chunks:
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )

    for chunk_text, item in zip(chunk, response.data):
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



print("Indexing Complete!")