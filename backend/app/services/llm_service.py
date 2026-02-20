import os
from groq import Groq
from dotenv import load_dotenv
from app.services.memory import add_message, get_conversation
from app.core.tone_manager import get_tone_config

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are AIRA, an emotionally intelligent and empathetic AI assistant.

Your personality:
- Soft, calm, and gentle tone
- Validate emotions before suggesting anything
- Never judge
- Never dismiss feelings
- Encourage gentle reflection
- Do NOT provide medical diagnosis
- If self-harm intent appears, encourage seeking real-world help calmly

Behavior Rules:
1. Always acknowledge the user's emotion first.
2. Speak naturally like a caring friend, not a therapist.
3. Keep responses short (2–5 lines).
4. Avoid long explanations.
5. Avoid sounding robotic or overly formal.
6. Do not overuse motivational quotes.
7. Ask soft follow-up questions when appropriate.

If user expresses serious self-harm intent:
- Stay calm
- Show concern
- Encourage contacting a trusted person
- Suggest professional help gently

You are emotional support, not a doctor.

You are aware that the app has the following features:
- Breathing Mode (guided breathing for anxiety and stress)
- Hydration Mode (reminds user to drink water)
- Grounding Techniques (5-4-3-2-1 calming exercises)
- Diary Mode (private journaling space)

IMPORTANT:
- Do NOT suggest these features every time.
- Only suggest them when they are clearly helpful.
- Suggest them naturally inside conversation.
- Do not sound robotic or like a menu.
- Make suggestions feel caring and intuitive.
"""


def generate_response(
    user_text: str,
    detected_emotion: str,
    emotion_history: list,
    user_id: str = "default"
):

    # Optional tone config (if used elsewhere)
    tone_config = get_tone_config(detected_emotion)

    # Save user message into memory
    add_message(user_id, "user", user_text)

    conversation_history = get_conversation(user_id)

    # -------------------------------
    # 🔥 Smart Feature Suggestion Layer
    # -------------------------------

    feature_hint = ""
    text_lower = user_text.lower()

    if any(word in text_lower for word in [
        "chest", "tight", "panic", "anxious", "anxiety",
        "overwhelmed", "can't breathe", "cant breathe"
    ]):
        feature_hint = "Breathing Mode may be helpful."

    elif any(word in text_lower for word in [
        "water", "dehydrated", "thirsty", "haven’t had water",
        "havent had water", "no water today"
    ]):
        feature_hint = "Hydration Mode may be helpful."

    elif any(word in text_lower for word in [
        "overthinking", "racing thoughts", "can't focus",
        "cant focus", "distracted", "mind is running"
    ]):
        feature_hint = "Grounding Techniques may be helpful."

    elif any(word in text_lower for word in [
        "write", "journal", "track my day", "diary",
        "start writing", "reflect"
    ]):
        feature_hint = "Diary Mode may be helpful."

    # -------------------------------
    # 🧠 Build LLM Messages
    # -------------------------------

    messages = [
        {
            "role": "system",
            "content": f"""
{SYSTEM_PROMPT}

Current detected emotion: {detected_emotion}
Emotion history: {emotion_history}

Potential helpful feature right now: {feature_hint}

If a feature seems helpful, suggest it gently and naturally.
If not necessary, continue normal emotional support.
"""
        }
    ]

    # Add past conversation memory
    messages.extend(conversation_history)

    # -------------------------------
    # 🤖 Call Groq LLM
    # -------------------------------

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )

    assistant_reply = completion.choices[0].message.content

    # Save assistant reply in memory
    add_message(user_id, "assistant", assistant_reply)

    return assistant_reply