"""
api/analyze.py
---------------------
Phase 6 – Multimodal Fusion Endpoint

Automatically:
• Accepts optional text
• Accepts optional face image
• Accepts optional voice audio
• Combines available modalities
"""

from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from app.services.text_emotion import analyze_text_emotion
from app.services.voice_emotion import analyze_voice
from app.services.face_emotion import analyze_face_emotion
from app.services.fusion import fuse_emotions


router = APIRouter(prefix="/analyze", tags=["Multimodal Fusion"])


@router.post("")
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    face: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
):
    """
    Accept any combination of:
    - text
    - face image
    - voice audio

    Fusion automatically works with available inputs.
    """

    text_scores = None
    face_scores = None
    voice_scores = None

    # ----------------------
    # TEXT
    # ----------------------
    if text:
        text_result = analyze_text_emotion(text)
        text_scores = {
            text_result["emotion"]: text_result["confidence"]
        }

    # ----------------------
    # FACE
    # ----------------------
    if face:
        face_bytes = await face.read()
        face_result = analyze_face_emotion(face_bytes)

        if "scores" in face_result:
            face_scores = face_result["scores"]
        else:
            face_scores = {
                face_result["emotion"]: face_result.get("confidence", 0)
            }

    # ----------------------
    # VOICE
    # ----------------------
    if voice:
        voice_bytes = await voice.read()
        voice_result = analyze_voice(voice_bytes)
        voice_scores = voice_result["all_scores"]

    # ----------------------
    # FUSION
    # ----------------------
    final_result = fuse_emotions(
        text_scores=text_scores,
        face_scores=face_scores,
        voice_scores=voice_scores,
    )

    return final_result
