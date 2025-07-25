from fastapi import APIRouter
from pydantic import BaseModel
import uuid
from app.services.agent import Chatbot
from app.config import PINECONE_API_KEY, OPENAI_API_KEY
import uuid
import base64
from typing import List

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

# Request model for evaluation
class EvalItem(BaseModel):
    question: str
    expected_answer: str

class EvalRequest(BaseModel):
    data: List[EvalItem]

# Route: POST /evaluate
@router.post("/evaluate")
def evaluate_rag(eval_request: EvalRequest):
    eval_data = [item.dict() for item in eval_request.data]

    results = []
    total_relevance = 0.0
    total_groundedness = 0.0

    def cosine_similarity(a, b):
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    for item in eval_data:
        question = item["question"]
        expected = item["expected_answer"]

        docs = bot.retriever.invoke(question)
        doc_texts = [doc.page_content for doc in docs]

        answer, _ = bot.get_answer(question, [])

        # Step 3: Get embeddings
        expected_emb = bot.embeddings.embed_query(expected)
        answer_emb = bot.embeddings.embed_query(answer)
        doc_embs = [bot.embeddings.embed_query(text) for text in doc_texts]

        # Step 4: Calculate scores
        relevance = cosine_similarity(expected_emb, answer_emb)
        groundedness = max(cosine_similarity(answer_emb, emb) for emb in doc_embs)

        total_relevance += relevance
        total_groundedness += groundedness

        results.append({
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "relevance_score": relevance,
            "groundedness_score": groundedness
        })

    avg_relevance = total_relevance / len(eval_data)
    avg_groundedness = total_groundedness / len(eval_data)

    return {
        "average_relevance": avg_relevance,
        "average_groundedness": avg_groundedness,
        "results": results
    }

