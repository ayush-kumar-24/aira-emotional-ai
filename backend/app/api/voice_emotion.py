"""
api/voice_emotion.py
---------------------
Voice Emotion Detection Endpoint
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.voice_emotion import analyze_voice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze-voice", tags=["Voice Emotion Analysis"])

# Max accepted file size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Accepted MIME types
ACCEPTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/webm",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "audio/aac",
    "application/octet-stream",
}


@router.post(
    "",
    summary="Detect emotion from voice/audio",
    response_description="Detected emotion with confidence and acoustic features",
)
async def analyze_voice_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file (wav / mp3 / ogg / webm / m4a / flac). Max 10 MB.",
    ),
):
    """
    Voice Emotion Detection

    Upload an audio file and receive the detected emotion.

    Supported formats:
    - WAV (recommended for best quality)
    - MP3, OGG, WebM, M4A, FLAC

    Detected Emotions:
    - sad: Low energy, monotone, slow speech
    - calm: Moderate steady energy, controlled
    - neutral: Average across all features
    - happy: Elevated pitch, bright timbre, higher energy
    - excited: Very high energy, dynamic, wide pitch range
    - angry: High energy, harsh timbre, tense
    - fearful: Unstable/trembling voice, erratic

    Tips for Best Results:
    - Use clear audio with minimal background noise
    - At least 1-2 seconds of speech
    - Natural speaking (not reading monotone text)
    - WAV format provides best accuracy
    """
    
    # Log incoming request
    logger.info(f"Received audio file: {file.filename} ({file.content_type})")
    
    # Validate content type (soft check)
    content_type = file.content_type or ""
    if content_type and content_type not in ACCEPTED_AUDIO_TYPES:
        logger.warning(f"Unexpected content type: {content_type}")
    
    # Read file bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=400, 
            detail="Could not read the uploaded file."
        )
    
    # Size validation
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Uploaded audio file is empty."
        )
    
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )
    
    logger.info(f"File size: {len(audio_bytes) / 1024:.2f} KB")
    
    # Run emotion analysis
    try:
        result = analyze_voice(audio_bytes)
        logger.info(f"Analysis complete: {result['emotion']} (confidence: {result['confidence']:.2f})")
        return JSONResponse(content=result)
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
        
    except RuntimeError as rte:
        logger.error(f"Runtime error: {rte}")
        raise HTTPException(status_code=500, detail=str(rte))
        
    except Exception as e:
        logger.exception("Unhandled error in voice emotion endpoint")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during voice analysis."
        )
