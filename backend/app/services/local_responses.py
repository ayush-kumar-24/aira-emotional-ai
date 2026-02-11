import random

RESPONSES = {
    "sadness": [
        "I’m really glad you shared that. You don’t have to go through it alone.",
        "Some days feel heavier than others — but you’re stronger than you think.",
        "Even quiet days pass. I’m here with you right now."
    ],
    "joy": [
        "That’s awesome to hear! Moments like these deserve a smile.",
        "I love that energy — keep it going!",
        "Sounds like a great day for you!"
    ],
    "anger": [
        "Take a slow breath — you’ve handled tough things before.",
        "It’s okay to pause and reset. You’re in control.",
        "Frustration happens, but it doesn’t define you."
    ],
    "fear": [
        "You’re safer than it feels right now. Take one step at a time.",
        "It’s okay to feel uncertain — you’re not alone.",
        "Breathe slowly. You’ve gotten through difficult moments before."
    ],
    "neutral": [
        "I’m here with you. What’s on your mind today?",
        "How’s your day going so far?",
        "Want to talk about anything?"
    ]
}

def generate_local_response(emotion: str):
    if emotion not in RESPONSES:
        emotion = "neutral"

    return random.choice(RESPONSES[emotion])
