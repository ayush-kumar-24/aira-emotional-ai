from fastapi import FastAPI
from app.api.emotion import router as emotion_router

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(emotion_router)
