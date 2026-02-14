"""
api/chat.py
-----------
Text-to-Text conversation (NO voice)
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import detect_crisis
from app.services.llm_service import generate_response
from app.services.memory import add_emotion, get_emotion_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Text Chat"])


class ChatInput(BaseModel):
    text: str


@router.post("", summary="Text-to-Text conversation")
async def chat(data: ChatInput):
    """
    Text Chat Mode
    
    User sends text → AI replies with text only
    No audio generation in this mode
    """
    
    logger.info(f"📝 Text mode - Received: {data.text[:50]}...")

    # 🚨 Crisis Check
    crisis_data = detect_crisis(data.text)
    if crisis_data["is_crisis"]:
        return {
            "emotion": "crisis",
            "response_text": crisis_data["message"]
        }

    # 🧠 Emotion Detection
    emotion_result = analyze_text_emotion(data.text)
    emotion = emotion_result["emotion"]

    add_emotion("demo_user", emotion)
    emotion_history = get_emotion_history("demo_user")

    # 🤖 LLM Response
    assistant_reply = generate_response(
        data.text,
        emotion,
        emotion_history,
        user_id="demo_user"
    )

    logger.info(f"✅ Text response generated")

    return {
        "emotion": emotion,
        "response_text": assistant_reply
        # NO audio in text mode!
    }