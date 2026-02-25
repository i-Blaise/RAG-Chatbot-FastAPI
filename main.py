from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from rag import chatbot
import os



app = FastAPI()

ADMIN_SECRET = os.getenv("ADMIN_SECRET")

class ChatRequest(BaseModel):
    message: str

class SourceChunk(BaseModel):
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        result = chatbot(request.message)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))