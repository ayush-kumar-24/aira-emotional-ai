"""
api/face_emotion.py
-------------------
Face emotion detection endpoint
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.face_emotion import analyze_face_emotion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face-emotion", tags=["Face Emotion Analysis"])

# Max accepted file size: 5 MB
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# Accepted image types
ACCEPTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


@router.post(
    "",
    summary="Detect emotion from facial image",
    response_description="Detected emotion with confidence from face"
)
async def face_emotion(
    file: UploadFile = File(
        ...,
        description="Image file with face (jpg / png / webp). Max 5 MB."
    )
):
    """
    Face Emotion Detection
    
    Upload an image containing a face and receive the detected emotion.
    
    Supported formats:
    - JPEG/JPG
    - PNG
    - WebP
    
    Detected Emotions:
    - sad, calm, neutral, happy, excited, angry, fearful
    
    Tips for Best Results:
    - Clear, well-lit face
    - Face should be clearly visible
    - Front-facing portrait works best
    - Single face per image recommended
    """
    
    logger.info(f"Received image file: {file.filename} ({file.content_type})")
    
    # Validate content type (soft check)
    content_type = file.content_type or ""
    if content_type and content_type not in ACCEPTED_IMAGE_TYPES:
        logger.warning(f"Unexpected content type: {content_type}")
    
    # Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file."
        )
    
    # Size validation
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image file is empty."
        )
    
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )
    
    logger.info(f"File size: {len(image_bytes) / 1024:.2f} KB")
    
    # Run emotion analysis
    try:
        result = analyze_face_emotion(image_bytes)
        
        logger.info(f"Analysis complete: {result.get('emotion')} (confidence: {result.get('confidence', 0):.2f})")
        
        return JSONResponse(content={
            "filename": file.filename,
            "emotion": result.get("emotion"),
            "confidence": result.get("confidence"),
            "all_scores": result.get("all_scores", {}),
            "face_detected": result.get("face_detected", True)
        })
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
        
    except Exception as e:
        logger.exception("Unhandled error in face emotion endpoint")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during face analysis."
        )
