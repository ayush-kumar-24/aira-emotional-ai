from fastapi import FastAPI
from app.api.emotion import router as emotion_router
from app.api.chat import router as chat_router

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(emotion_router)
app.include_router(chat_router)