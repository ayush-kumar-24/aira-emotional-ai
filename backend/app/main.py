"""
main.py
-------
AIRA Emotional AI - Main Application
"""

from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from app.api.chat import router as chat_router
app = FastAPI(
    title="AIRA Emotional AI",
    version="1.0.0",
    description="Multimodal emotion detection from text, voice, and face",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(chat_router)

from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router
from app.api.voice_emotion import router as voice_emotion_router
from app.api.face_emotion import router as face_emotion_router
from app.api.analyze import router as analyze_router

from app.db.database import engine
from app.db. models import Base

Base.metadata.create_all(bind=engine)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES - NO TAGS HERE! Let routers define their own tags
# ============================================================

app.include_router(health_router)
app.include_router(analyze_router)

app.include_router(voice_emotion_router)
app.include_router(face_emotion_router)


@app.get("/", tags=["Info"])
async def root():
    """API Information"""
    return {
        "message": "AIRA Emotional AI",
        "version": "1.0.0",
        "documentation": "/docs",
        "main_endpoint": "/analyze"
    }

