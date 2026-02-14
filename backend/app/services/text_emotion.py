"""
services/text_emotion.py
------------------------
Text Emotion Detection with SSL Fix
"""

# ====== SSL FIX - SABSE PEHLE ======
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
print("✅ SSL fix applied for text emotion model...")
# ====== SSL FIX END ======


from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Load model with error handling
try:
    print("📥 Downloading/Loading emotion model from HuggingFace...")
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    print("✅ Emotion model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load emotion model: {e}")
    emotion_pipeline = None


def analyze_text_emotion(text: str):
    """Analyze emotion from text"""
    
    # Validation
    if not text or text.strip() == "":
        return {"emotion": "neutral", "confidence": 0.0}
    
    # Fallback if model failed to load
    if emotion_pipeline is None:
        logger.warning("Model not available, returning neutral emotion")
        return {"emotion": "neutral", "confidence": 0.5}
    
    try:
        results = emotion_pipeline(text)[0]
        best = max(results, key=lambda x: x["score"])
        
        return {
            "emotion": best["label"],
            "confidence": round(best["score"], 3)
        }
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return {"emotion": "neutral", "confidence": 0.0}