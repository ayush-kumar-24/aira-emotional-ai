from app.core.constants import CRISIS_KEYWORDS
from app.services.local_responses import generate_crisis_response

def detect_crisis(text: str):
    text_lower = text.lower()

    for word in CRISIS_KEYWORDS:
        if word in text_lower:
            return {
                "is_crisis": True,
                "level": "high",
                "message": generate_crisis_response()
            }

    return {
        "is_crisis": False,
        "level": "none"
    }
