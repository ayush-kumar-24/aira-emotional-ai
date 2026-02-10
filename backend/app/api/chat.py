from fastapi import APIRouter
from pydantic import BaseModel
from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import detect_crisis

router = APIRouter(prefix="/api", tags=["chat"])

class TextInput(BaseModel):
    text: str

@router.post("/analyze-text")
def analyze_text(data: TextInput):
    return analyze_text_emotion(data.text)

@router.post("/crisis-detection")
def detect_crisis_endpoint(data: TextInput):
    return detect_crisis(data.text)