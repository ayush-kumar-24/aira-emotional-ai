from fastapi import APIRouter
from pydantic import BaseModel
from app.services.text_emotion import analyze_text_emotion

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/analyze-text")
def analyze_text(data: TextInput):
    return analyze_text_emotion(data.text)
