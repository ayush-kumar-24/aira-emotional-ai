"""
api/analyze.py
--------------
Orchestrator Endpoint - Multimodal Emotion Analysis
Automatically detects available inputs and fuses results
"""

import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.services.text_emotion import analyze_text_emotion
from app.services.voice_emotion import analyze_voice
from app.services.face_emotion import analyze_face_emotion
from app.services.fusion import fuse_emotions, get_emotion_explanation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Multimodal Analysis"])

# File size limits
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024   # 5 MB


@router.post(
    "",
    summary="Multimodal Emotion Analysis (Text + Voice + Face)",
    response_description="Unified emotion analysis across all available modalities",
)
async def analyze_multimodal(
    text: Optional[str] = Form(None, description="Text input for emotion analysis"),
    audio: Optional[UploadFile] = File(None, description="Audio file (wav/mp3/ogg/webm/m4a)"),
    image: Optional[UploadFile] = File(None, description="Image file (jpg/png) with face"),
):
    """
    ## 🧠 Multimodal Emotion Analysis
    
    The orchestrator endpoint that automatically:
    1. Detects what inputs you provide (text, audio, image)
    2. Analyzes each available modality
    3. Intelligently fuses the results
    4. Returns unified emotion with confidence
    
    ### Input Options (all optional, but provide at least one):
    - **text**: Any text message or statement
    - **audio**: Voice recording (wav, mp3, ogg, webm, m4a) - max 10MB
    - **image**: Photo with face (jpg, png) - max 5MB
    
    ### Response Structure:
```json
    {
      "emotion": "happy",
      "confidence": 0.82,
      "explanation": "Joyful, content, positive",
      "modalities_used": ["text", "voice", "face"],
      "individual_results": {
        "text": {"emotion": "happy", "confidence": 0.75},
        "voice": {"emotion": "happy", "confidence": 0.85},
        "face": {"emotion": "happy", "confidence": 0.90}
      },
      "fused_scores": {
        "sad": 0.05,
        "calm": 0.10,
        "neutral": 0.15,
        "happy": 0.82,
        "excited": 0.45,
        "angry": 0.02,
        "fearful": 0.03
      },
      "fusion_method": "weighted_average",
      "conflict_detected": false
    }
```
    
    ### Fusion Weights:
    - Text: 1.0x
    - Voice: 1.2x (slightly more reliable)
    - Face: 1.3x (most reliable for emotions)
    
    ### Example Usage:
    
    **Text only:**
```
    text: "I'm having a great day!"
    → Returns text emotion analysis
```
    
    **Voice only:**
```
    audio: voice_recording.wav
    → Returns voice emotion analysis
```
    
    **Text + Voice:**
```
    text: "I'm fine..."
    audio: sad_voice.wav
    → Fuses both (might detect sadness despite text)
```
    
    **All three:**
```
    text: "I'm happy"
    audio: happy_voice.wav
    image: smiling_face.jpg
    → Most accurate - uses all available signals
```
    """
    
    logger.info("=" * 60)
    logger.info("🎯 NEW MULTIMODAL ANALYSIS REQUEST")
    logger.info("=" * 60)
    
    # Check if at least one input is provided
    if not any([text, audio, image]):
        raise HTTPException(
            status_code=400,
            detail="At least one input required: text, audio, or image"
        )
    
    # Track what's available
    available = []
    if text:
        available.append("text")
    if audio:
        available.append("audio")
    if image:
        available.append("image")
    
    logger.info(f"📥 Inputs received: {', '.join(available)}")
    
    # === ANALYZE TEXT ===
    text_scores = None
    if text:
        try:
            logger.info("📝 Analyzing text...")
            text_result = analyze_text_emotion(text)
            text_scores = text_result.get("all_scores")
            logger.info(f"✅ Text analysis complete: {text_result['emotion']} ({text_result['confidence']:.2f})")
        except Exception as e:
            logger.error(f"❌ Text analysis failed: {e}")
            # Continue with other modalities
    
    # === ANALYZE VOICE ===
    voice_scores = None
    if audio:
        try:
            logger.info(f"🎤 Analyzing voice: {audio.filename}")
            
            # Read audio bytes
            audio_bytes = await audio.read()
            
            # Size check
            if len(audio_bytes) == 0:
                logger.warning("⚠️  Empty audio file")
            elif len(audio_bytes) > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large. Max {MAX_AUDIO_SIZE // (1024*1024)} MB"
                )
            else:
                voice_result = analyze_voice(audio_bytes)
                voice_scores = voice_result.get("all_scores")
                logger.info(f"✅ Voice analysis complete: {voice_result['emotion']} ({voice_result['confidence']:.2f})")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Voice analysis failed: {e}")
            # Continue with other modalities
    
    # === ANALYZE FACE ===
    face_scores = None
    if image:
        try:
            logger.info(f"😊 Analyzing face: {image.filename}")
            
            # Read image bytes
            image_bytes = await image.read()
            
            # Size check
            if len(image_bytes) == 0:
                logger.warning("⚠️  Empty image file")
            elif len(image_bytes) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image file too large. Max {MAX_IMAGE_SIZE // (1024*1024)} MB"
                )
            else:
                face_result = analyze_face_emotion(image_bytes)
                face_scores = face_result.get("all_scores")
                logger.info(f"✅ Face analysis complete: {face_result['emotion']} ({face_result['confidence']:.2f})")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Face analysis failed: {e}")
            # Continue with other modalities
    
    # === FUSE RESULTS ===
    logger.info("🔄 Fusing emotions from all available modalities...")
    
    fusion_result = fuse_emotions(
        text_scores=text_scores,
        voice_scores=voice_scores,
        face_scores=face_scores
    )
    
    # Add explanation
    fusion_result["explanation"] = get_emotion_explanation(fusion_result["emotion"])
    
    logger.info("=" * 60)
    logger.info(f"🎉 FINAL RESULT: {fusion_result['emotion'].upper()} (confidence: {fusion_result['confidence']:.2f})")
    logger.info("=" * 60)
    
    return JSONResponse(content=fusion_result)
