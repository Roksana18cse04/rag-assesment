from fastapi import FastAPI
from app.routes.chatbot_routes import router 


app = FastAPI(title="Answer Generator")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Answer Generator API!"}

app.include_router(router, prefix="/QA-chatbot", tags=["question-answering-chatbot"])

