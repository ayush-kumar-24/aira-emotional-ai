"""
api/chat.py
-----------
Text emotion analysis endpoint
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import detect_crisis
from app.services.llm_service import generate_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Text Emotion Analysis"])


class ChatInput(BaseModel):
    text: str


@router.post(
    "",
    summary="Analyze emotion from text",
    response_description="Emotion analysis with empathetic response"
)
async def chat(data: ChatInput):

    logger.info(f"Received text: {data.text[:50]}...")

    # 🚨 Crisis Check First
    crisis_data = detect_crisis(data.text)

    if crisis_data["is_crisis"]:
        logger.warning("Crisis detected.")
        return {
            "emotion": "crisis",
            "response": crisis_data["message"]
        }

    # 🧠 Emotion Analysis
    emotion_result = analyze_text_emotion(data.text)
    emotion = emotion_result["emotion"]

    logger.info(f"Detected emotion: {emotion}")

    # 🤖 LLM Response
    llm_reply = generate_response(data.text, emotion)

    return {
        "emotion": emotion,
        "response": llm_reply
    }
