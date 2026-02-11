"""
main.py
-------
AIRA Emotional AI - Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.face_emotion import router as face_emotion_router
from app.api.voice_emotion import router as voice_emotion_router
from app.api.fusion import router as fusion_router
from app.api.analyze import router as analyze_router


app = FastAPI(
    title="AIRA Emotional AI",
    version="0.1.0",
    description="AI-powered emotion detection from text, face, and voice",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (if you need to call from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core routes
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, tags=["Chat / Text Emotion"])
app.include_router(face_emotion_router, tags=["Face Emotion"])
app.include_router(voice_emotion_router, tags=["Voice Emotion"])
app.include_router(fusion_router, tags=["Fusion"])
app.include_router(analyze_router)






@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Welcome to AIRA Emotional AI API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "text_emotion": "/chat",
            "face_emotion": "/analyze-face",
            "voice_emotion": "/analyze-voice"
        }
    }