"""
api/fusion.py
-------------
Phase 6 – Multimodal Fusion API Endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from app.services.fusion import fuse_emotions


router = APIRouter(prefix="/fusion", tags=["Fusion"])


# Request Schema
class FusionRequest(BaseModel):
    text_scores: Optional[Dict[str, float]] = None
    voice_scores: Optional[Dict[str, float]] = None
    face_scores: Optional[Dict[str, float]] = None


@router.post("", summary="Fuse multimodal emotion scores")
def fusion_endpoint(request: FusionRequest):
    """
    Accepts optional emotion score dictionaries from:
    - Text
    - Voice
    - Face

    Returns final fused emotion.
    """

    try:
        result = fuse_emotions(
            text_scores=request.text_scores,
            voice_scores=request.voice_scores,
            face_scores=request.face_scores,
        )

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Fusion processing failed."
        )
