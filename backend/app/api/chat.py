"""
api/chat.py
-----------
Text emotion analysis endpoint
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import check_crisis
from app.core.constants import EMPATHY_RESPONSES

logger = logging.getLogger(__name__)

# Clean router definition
router = APIRouter(prefix="/api/chat", tags=["Text Emotion Analysis"])


class ChatInput(BaseModel):
    """Input model for chat endpoint"""
    text: str


@router.post(
    "",
    summary="Analyze emotion from text",
    response_description="Emotion analysis with empathetic response"
)
async def chat(data: ChatInput):
    """
    Analyze emotion from text input and provide empathetic response.
    
    - Checks for crisis keywords first
    - Analyzes emotion if no crisis detected
    - Returns appropriate empathetic response
    """
    logger.info(f"Received text: {data.text[:50]}...")
    
    # Crisis check
    crisis_result = check_crisis(data.text)
    if crisis_result["crisis"]:
        logger.warning("Crisis detected in text")
        return crisis_result

    # Emotion analysis
    emotion_result = analyze_text_emotion(data.text)
    emotion = emotion_result["emotion"]
    
    logger.info(f"Detected emotion: {emotion}")

    # Get empathetic response
    response = EMPATHY_RESPONSES.get(
        emotion,
        "I'm here for you. Tell me more."
    )

    return {
        "user_text": data.text,
        "emotion": emotion,
        "confidence": emotion_result.get("confidence", 0.0),
        "all_scores": emotion_result.get("all_scores", {}),
        "response": response
    }