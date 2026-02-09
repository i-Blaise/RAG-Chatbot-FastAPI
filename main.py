from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from rag import chatbot
import os



app = FastAPI()

ADMIN_SECRET = os.getenv("ADMIN_SECRET")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str




@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        answer = chatbot(request.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))