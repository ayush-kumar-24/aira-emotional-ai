"""
api/voice_emotion.py
---------------------
Voice-to-Voice conversation (with text for history)
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.voice_emotion import analyze_voice
from app.services.text_emotion import analyze_text_emotion
from app.services.crisis import detect_crisis
from app.services.llm_service import generate_response
from app.services.memory import add_emotion, get_emotion_history
from app.services.tts_service import generate_audio

from app.db.database import SessionLocal
from app.db.models import Conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze-voice", tags=["Voice Chat"])

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

ACCEPTED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm", "audio/mp4", "audio/m4a",
    "audio/x-m4a", "audio/flac", "audio/aac",
    "application/octet-stream",
}


@router.post("", summary="Voice-to-Voice conversation")
async def analyze_voice_endpoint(
    file: UploadFile = File(..., description="Audio file. Max 10 MB."),
):
    """
    Voice Chat Mode
    
    User speaks → AI replies with voice + text
    """

    logger.info(f"🎤 Voice mode - Received audio: {file.filename}")

    # Validate content type
    content_type = file.content_type or ""
    if content_type and content_type not in ACCEPTED_AUDIO_TYPES:
        logger.warning(f"Unexpected content type: {content_type}")

    # Read file
    try:
        audio_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Max 10 MB"
        )

    # ========================================
    # STEP 1: Voice Analysis (STT + Emotion)
    # ========================================
    try:
        voice_result = analyze_voice(audio_bytes)
        transcription = voice_result.get("transcription", "")
        voice_emotion = voice_result["emotion"]
    except Exception:
        raise HTTPException(status_code=500, detail="Voice analysis failed")

    if not transcription.strip():
        return JSONResponse(content={
            "transcription": "",
            "voice_emotion": voice_emotion,
            "confidence": voice_result.get("confidence"),
            "response_text": "I couldn't hear what you said. Could you try again?",
            "response_audio_url": None
        })

    # ========================================
    # STEP 2: Crisis Check
    # ========================================
    crisis_data = detect_crisis(transcription)

    if crisis_data["is_crisis"]:
        crisis_audio_url = generate_audio(crisis_data["message"])

        return JSONResponse(content={
            "transcription": transcription,
            "voice_emotion": voice_emotion,
            "emotion": "crisis",
            "response_text": crisis_data["message"],
            "response_audio_url": crisis_audio_url,
            "is_crisis": True
        })

    # ========================================
    # STEP 3: Emotion + LLM Response
    # ========================================
    try:
        text_emotion_result = analyze_text_emotion(transcription)
        text_emotion = text_emotion_result["emotion"]

        final_emotion = voice_emotion

        add_emotion("demo_user", final_emotion)
        emotion_history = get_emotion_history("demo_user")

        llm_response = generate_response(
            transcription,
            final_emotion,
            emotion_history,
            user_id="demo_user"
        )

    except Exception:
        llm_response = "I heard you, but I'm having trouble responding right now."
        text_emotion = "neutral"

    # ========================================
    # STEP 4: Generate Voice Response
    # ========================================
    try:
        response_audio_url = generate_audio(llm_response)
    except Exception:
        response_audio_url = None

    # ========================================
    # STEP 5: Save Conversation to SQLite
    # ========================================
    try:
        db = SessionLocal()
        conversation = Conversation(
            user_message=transcription,
            assistant_message=llm_response,
            emotion=voice_emotion
        )
        db.add(conversation)
        db.commit()
        db.close()
        logger.info("💾 Conversation saved to DB")
    except Exception as e:
        logger.error(f"Database save failed: {e}")

    # ========================================
    # FINAL RESPONSE
    # ========================================
    return JSONResponse(content={
        "transcription": transcription,
        "voice_emotion": voice_emotion,
        "text_emotion": text_emotion,
        "confidence": voice_result.get("confidence"),
        "response_text": llm_response,
        "response_audio_url": response_audio_url,
        "features": voice_result.get("features"),
        "all_scores": voice_result.get("all_scores")
    })