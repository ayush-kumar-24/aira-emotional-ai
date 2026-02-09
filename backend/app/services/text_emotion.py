from transformers import pipeline

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def analyze_text_emotion(text: str):
    if not text or text.strip() == "":
        return {"emotion": "neutral", "confidence": 0.0}

    results = emotion_pipeline(text)[0]
    best = max(results, key=lambda x: x["score"])

    return {
        "emotion": best["label"],
        "confidence": round(best["score"], 3)
    }
