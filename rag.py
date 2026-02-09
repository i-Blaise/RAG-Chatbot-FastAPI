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

    index = faiss.read_index("data/faiss.index")
    with open("data/metadata", "rb") as f:
        metadata = pickle.load(f)

    try:
        with open("chat_history.pkl", "rb") as f:
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

    valid_results = []

    for score, ids in zip(scores[0], indeces[0]):
        if score >= threshold:
            valid_results.append(score, metadata[ids])

    retrieved_chunks = [
        metadata[ids]
        for ids in indeces[0]
        if ids in metadata
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

    Answer using the context. If not available or you cant find the answer in the context, say 'Chief ano know, ano go lie'.

    User question:
    {query}
    """

    result = client.responses.create(
        input=query,
        model="gpt-5.2"
    )
    
    answer = result.output_text

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})


    try:
        with open("data/chat_history", "wb") as f:
            pickle.dump(chat_history, f)
    except FileNotFoundError:
        with open("data/chat_history", "wb") as f:
            pickle.dump(chat_history, f)

    return answer
