def get_tone_config(emotion: str):

    tone_map = {
        "sadness": {
            "style": "very gentle, slow, warm, comforting",
            "instructions": "Use soft language. Avoid giving too many solutions immediately."
        },
        "anger": {
            "style": "calm, grounding, stabilizing",
            "instructions": "Help them slow down. Encourage breathing and reflection."
        },
        "fear": {
            "style": "reassuring and stabilizing",
            "instructions": "Provide reassurance and reduce uncertainty."
        },
        "joy": {
            "style": "light, positive, encouraging",
            "instructions": "Match their energy while staying emotionally intelligent."
        },
        "neutral": {
            "style": "balanced and supportive",
            "instructions": "Keep tone conversational and warm."
        }
    }

    return tone_map.get(emotion, tone_map["neutral"])
