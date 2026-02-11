from fastapi import APIRouter
from pydantic import BaseModel
from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import check_crisis
from app.core.constants import EMPATHY_RESPONSES

router = APIRouter(prefix="/api", tags=["chat"])

class ChatInput(BaseModel):
    text: str

@router.post("/chat")
def chat(data: ChatInput):
    crisis_result = check_crisis(data.text)

    if crisis_result["crisis"]:
        return crisis_result

    emotion_result = analyze_text_emotion(data.text)
    emotion = emotion_result["emotion"]

    response = EMPATHY_RESPONSES.get(
        emotion,
        "I'm here for you. Tell me more."
    )

    return {
        "user_text": data.text,
        "emotion": emotion,
        "response": response
    }
