from collections import defaultdict

# Conversation history
conversation_memory = defaultdict(list)

# Emotion history
emotion_memory = defaultdict(list)


def add_message(user_id: str, role: str, content: str):
    conversation_memory[user_id].append({
        "role": role,
        "content": content
    })

    # Keep last 6 messages only
    conversation_memory[user_id] = conversation_memory[user_id][-6:]


def get_conversation(user_id: str):
    return conversation_memory[user_id]


def add_emotion(user_id: str, emotion: str):
    emotion_memory[user_id].append(emotion)

    # Keep last 10 emotions
    emotion_memory[user_id] = emotion_memory[user_id][-10:]


def get_emotion_history(user_id: str):
    return emotion_memory[user_id]