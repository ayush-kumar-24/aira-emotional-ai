import random

CRISIS_TEMPLATES = [
    """I’m really concerned about what you just shared. It sounds deeply painful, and I’m glad you reached out instead of staying silent. When thoughts feel this heavy, it can seem like there’s no way forward — but feelings can shift with the right support. If you are in immediate danger, please contact your local emergency number right now.You deserve real help and real care. Would you consider reaching out to someone you trust or a crisis helpline near you?""",
    """Reading your message makes me pause, because it sounds incredibly overwhelming. Moments like this can distort everything and make it feel unbearable — but you are not alone in this. If you’re at risk of harming yourself, please call emergency services immediately. I’m here with you. What’s making it feel especially intense right now?""",
    """I’m really glad you told me this. That takes strength, even if it doesn’t feel like it. When things feel this dark, support matters more than anything else. If you're in immediate danger, please reach out to emergency services or a crisis helpline in your area right now. You matter more than this moment. Tell me what’s been building up inside."""
]

def generate_local_response(emotion: str):
    if emotion not in RESPONSES:
        emotion = "neutral"

    return random.choice(RESPONSES[emotion])


def generate_crisis_response():
    return random.choice(CRISIS_TEMPLATES)