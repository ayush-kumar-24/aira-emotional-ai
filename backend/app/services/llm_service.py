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
You are AIRA, an emotional AI assistant.

You are aware that the app includes the following built-in wellness tools:
- Breathing Mode (guided breathing exercises)
- Hydration Mode (reminders to drink water)
- Grounding Techniques (5-4-3-2-1 method etc.)
- Diary Mode (private journaling space)

IMPORTANT:
- Do NOT always suggest these.
- Only suggest them when emotionally appropriate.
- Suggest gently, not forcefully.
- Never mention "frontend" or "feature" technically.
- Suggest naturally like a caring assistant."""


def generate_response(
    user_text: str,
    detected_emotion: str,
    emotion_history: list,
    user_id: str = "default"
):

    # Get tone configuration
    tone_config = get_tone_config(detected_emotion)

    # Add user message to memory
    add_message(user_id, "user", user_text)

    conversation_history = get_conversation(user_id)

    messages = [
        {
    "role": "system",
    "content": f"""
{SYSTEM_PROMPT}

Current detected emotion: {detected_emotion}
Emotion history: {emotion_history}

If repeated negative emotions appear, gently acknowledge the pattern.
"""
}
    ]

    # Add past conversation
    messages.extend(conversation_history)

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