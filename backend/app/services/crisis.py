from app.core.constants import CRISIS_KEYWORDS, HELPLINE_MESSAGE

def check_crisis(text: str):
    text_lower = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in text_lower:
            return {
                "crisis": True,
                "message": HELPLINE_MESSAGE
            }
    return {"crisis": False}
