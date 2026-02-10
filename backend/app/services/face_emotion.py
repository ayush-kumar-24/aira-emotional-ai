import numpy as np
from fer import FER
from PIL import Image
import io

# Load model ONCE (very important)
detector = FER(mtcnn=True)

def analyze_face_emotion(image_bytes: bytes):
    try:
        # Convert bytes → PIL → NumPy
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)

        # Detect emotions
        results = detector.detect_emotions(frame)

        if not results:
            return {
                "emotion": "no_face_detected",
                "confidence": 0.0,
                "scores": {}
            }

        emotions = results[0]["emotions"]

        # Get top emotion + score
        emotion, score = max(emotions.items(), key=lambda x: x[1])

        return {
            "emotion": emotion,
            "confidence": round(float(score), 3),
            "scores": {k: round(float(v), 3) for k, v in emotions.items()}
        }

    except Exception as e:
        return {
            "error": str(e)
        }