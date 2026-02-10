from fastapi import APIRouter, UploadFile, File
from app.services.face_emotion import analyze_face_emotion

router = APIRouter(tags=["Face Emotion"])

@router.post("/face-emotion")
async def face_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = analyze_face_emotion(image_bytes)

    return {
        "filename": file.filename,
        "emotion": result.get("emotion"),
        "confidence": result.get("confidence")
    }
