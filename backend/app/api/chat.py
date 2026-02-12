"""
api/chat.py
-----------
Text emotion analysis endpoint
"""

from email.mime import text
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import check_crisis
from app.core.constants import EMPATHY_RESPONSES
from app.services.llm_service import generate_response


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
    llm_reply = generate_response(data.text, emotion)

    return {
    "emotion": emotion,
    "response": llm_reply
}
