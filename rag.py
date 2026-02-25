import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import faiss
import pickle

load_dotenv()

def chatbot(message: str):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    index_path = "data/faiss.index"
    metadata_path = "data/metadata.pkl"
    if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            "RAG index not found. Run build_index.py first (requires PDF in kb/)."
        )
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    try:
        with open("data/chat_history.pkl", "rb") as f:
            chat_history = pickle.load(f)
    except FileNotFoundError:
        chat_history = []
    
    query = message.strip()

    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    faiss.normalize_L2(query_vector)
    

    k = 3
    threshold = 0.57

    scores, indeces = index.search(query_vector, k)

    sources = []
    for score, idx in zip(scores[0], indeces[0]):
        if idx in metadata:
            sources.append({"text": metadata[idx], "score": float(score)})

    retrieved_chunks = [
        metadata[idx]
        for idx in indeces[0]
        if idx in metadata
    ]

    history_chat = ""
    for turn in chat_history[-5:]:
        history_chat += f"{turn['role']}: {turn['content']}\n"
    

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a helpful assistant named Mega.

    Conversation history:
    {history_chat}

    Context:
    {context}

    Answer using the context. If not available or you cant find the answer in the context, say "I don't have information about that in the provided Knowledge base."

    User question:
    {query}
    """

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = result.choices[0].message.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})


    os.makedirs("data", exist_ok=True)
    with open("data/chat_history.pkl", "wb") as f:
        pickle.dump(chat_history, f)

    return {"answer": answer, "sources": sources}
