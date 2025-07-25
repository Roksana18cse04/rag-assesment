from fastapi import APIRouter
from pydantic import BaseModel
import uuid
from app.services.agent import Chatbot
from app.config import PINECONE_API_KEY, OPENAI_API_KEY
import uuid
import base64

router = APIRouter()

def generate_short_session_id():
    uid = uuid.uuid4()
    short_id = base64.urlsafe_b64encode(uid.bytes).decode('utf-8').rstrip("=")
    return short_id

# In-memory short-term session memory
session_histories = {}

bot = Chatbot(
    pinecone_api_key=PINECONE_API_KEY,
    openai_api_key=OPENAI_API_KEY,
)

# Start new chat session
@router.get("/start-chat")
def start_chat():
    session_id = generate_short_session_id()
    session_histories[session_id] = []
    return {"session_id": session_id, "message": "New chat session started."}

@router.delete("/end-chat/{session_id}")
def end_chat(session_id: str):
    session_histories.pop(session_id, None)
    return {"message": f"Session {session_id} ended and memory cleared."}

# Request model for /ask
class QuestionRequest(BaseModel):
    session_id: str
    question: str

# Ask a question (with session_id)
@router.post("/ask")
def ask_question(req: QuestionRequest):
    history = session_histories.get(req.session_id)
    if history is None:
        return {"error": "Invalid session_id. Please start a new chat."}

    answer, updated_history = bot.get_answer(req.question, history)
    session_histories[req.session_id] = updated_history
    print("session_histories", session_histories)
    return {
        "session_id": req.session_id,
        "question": req.question,
        "answer": answer
    }

# Get chat history for a session
@router.post("/history")
def get_history(session_id: str):
    history = session_histories.get(session_id)
    if history is None:
        return {"error": "No chat history found for this session."}

    return {
        "session_id": session_id,
        "history": history  # already a list of {"question": ..., "answer": ...}
    }
