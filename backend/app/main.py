from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.face_emotion import router as face_emotion_router

app = FastAPI(
    title="AIRA Emotional AI",
    version="0.1.0"
)

# Core routes
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, tags=["Chat / Text Emotion"])
app.include_router(face_emotion_router, tags=["Face Emotion"])
