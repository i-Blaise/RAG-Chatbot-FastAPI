# CyberLaw Assistant — Production-Grade RAG Chatbot (FastAPI + FAISS)

Live Demo: https://chat.artfricastudio.com/

A production-ready Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, FAISS, OpenAI embeddings, and React. The system enables users to query and analyze the Ghana Cybersecurity Amendment Bill 2025 with accurate, grounded responses based strictly on document content.

This project demonstrates a complete end-to-end RAG pipeline including document ingestion, semantic retrieval, persistent vector indexing, conversational context management, and source attribution.

---

## Key Features

- Persistent FAISS vector index (no re-embedding on startup)
- Token-based document chunking with overlap
- Semantic search using OpenAI text-embedding-3-small
- Conversational memory with contextual follow-up support
- Source attribution with similarity scores
- Production-ready FastAPI backend
- Fully deployed live system
- Retrieval transparency (view retrieved chunks and relevance)

---

## Architecture Overview

The system follows a standard production RAG architecture:

User Query  
↓  
FastAPI API Endpoint  
↓  
Embedding Generation (OpenAI)  
↓  
FAISS Vector Search  
↓  
Top-K Chunk Retrieval  
↓  
Prompt Assembly (with chat history)  
↓  
LLM Response Generation  
↓  
Return Answer + Sources + Relevance Scores  

---

## Project Structure

```
RAG-CHATBOT-FASTAPI/
│
├── data/                # Persistent FAISS index and metadata
├── kb/                  # Knowledge base documents
│
├── build_index.py       # Document ingestion and vector index builder
├── main.py              # FastAPI server and API endpoints
├── rag.py               # Core RAG pipeline (retrieval, prompting, memory)
├── requirements.txt     # Python dependencies
│
├── .env                 # Environment variables (not committed)
├── .gitignore
│
└── venv/                # Virtual environment (not committed)
```

---

## Core Components

### Document Ingestion (`build_index.py`)

Responsible for:

- Loading document(s) from `/kb`
- Token-based chunking with overlap
- Generating embeddings using OpenAI API
- Creating persistent FAISS index
- Saving index and metadata to `/data`

This step only needs to run when documents are added or updated.

---

### RAG Pipeline (`rag.py`)

Handles:

- Query embedding generation
- Semantic similarity search via FAISS
- Retrieval of top-K relevant chunks
- Chat history management
- Prompt construction
- LLM response generation
- Returning source attribution and similarity scores

---

### FastAPI Backend (`main.py`)

Provides REST API endpoint for frontend interaction.

Endpoint:

POST /chat

Request example:

```json
{
  "message": "What does the bill say about Artificial Intelligence?",
  "session_id": "optional-session-id"
}
```

Response example:

```json
{
  "answer": "AI is defined as...",
  "sources": [
    {
      "text": "...",
      "score": 0.51
    }
  ]
}
```

---

## Technologies Used

Backend:

- FastAPI
- Python
- FAISS (Vector Database)
- OpenAI Embeddings API
- OpenAI LLM API

Frontend:

- React

Infrastructure:

- VPS deployment
- Persistent vector storage

---

## Installation and Setup

Clone repository:

```
git clone https://github.com/yourusername/rag-chatbot-fastapi.git
cd rag-chatbot-fastapi
```

Create virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Build Vector Index

Run ingestion pipeline:

```
python build_index.py
```

This generates the FAISS index in `/data`.

---

## Run Server

Start FastAPI server:

```
uvicorn main:app --reload
```

Server runs at:

```
http://localhost:8000
```

---

## Deployment

The system is deployed and publicly accessible at:

https://chat.artfricastudio.com/

Deployment includes:

- FastAPI backend
- Persistent FAISS vector index
- React frontend
- Production environment configuration

---

## Design Decisions

### Why FAISS instead of managed vector database?

FAISS provides:

- High performance similarity search
- Full control over storage and retrieval
- No external dependency
- Lower latency and cost

---

### Why text-embedding-3-small?

Chosen for optimal balance between:

- Cost efficiency
- High semantic accuracy
- Fast embedding generation

---

### Why token-based chunking?

Ensures:

- Proper alignment with LLM token limits
- Better semantic coherence
- Improved retrieval accuracy

---

### Why persistent vector storage?

Prevents re-embedding documents on startup, reducing:

- API costs
- Startup latency
- Compute overhead

---

## Limitations

- Single-document knowledge base (current version)
- In-memory session storage
- No authentication layer
- No distributed scaling (current version)

---

## Future Improvements

- Multi-document support
- Redis-based session memory
- Streaming responses
- LangChain integration layer
- Evaluation framework for retrieval accuracy
- Distributed vector storage

---

## What This Project Demonstrates

This project demonstrates real-world AI engineering capabilities including:

- Production RAG system design
- Vector database implementation
- Semantic retrieval pipelines
- LLM system integration
- Backend API design
- Deployment and infrastructure management

---

## Author

Built and deployed by:

Blaise Sonzie Mennia

AI Engineer specializing in:

- Retrieval-Augmented Generation (RAG)
- LLM infrastructure
- AI-powered backend systems
- Production AI deployment

---

## License

MIT License
