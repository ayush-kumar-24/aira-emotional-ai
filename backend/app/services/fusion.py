"""
services/fusion.py
------------------
Phase 6 – Multimodal Emotion Fusion Engine

• Accepts optional text, voice, face emotion scores
• Dynamically normalizes weights
• Works with 1, 2, or 3 modalities
• Returns final fused emotion + confidence
"""

import logging

logger = logging.getLogger(__name__)

# Default weights (can be tuned later)
DEFAULT_WEIGHTS = {
    "text": 0.4,
    "voice": 0.3,
    "face": 0.3,
}

ALL_EMOTIONS = [
    "sad",
    "calm",
    "neutral",
    "happy",
    "excited",
    "angry",
    "fearful",
]


def _normalize_weights(active_modalities: list[str]) -> dict:
    """
    Normalize weights only for modalities that are present.
    """
    total = sum(DEFAULT_WEIGHTS[m] for m in active_modalities)

    return {
        m: DEFAULT_WEIGHTS[m] / total
        for m in active_modalities
    }


def fuse_emotions(
    text_scores: dict | None = None,
    voice_scores: dict | None = None,
    face_scores: dict | None = None,
) -> dict:
    """
    Main fusion function.

    Parameters:
        text_scores  -> dict of emotion: score
        voice_scores -> dict of emotion: score
        face_scores  -> dict of emotion: score

    Returns:
        {
            "final_emotion": str,
            "confidence": float,
            "fusion_scores": dict,
            "modalities_used": list
        }
    """

    # Determine active modalities
    active_modalities = []
    if text_scores:
        active_modalities.append("text")
    if voice_scores:
        active_modalities.append("voice")
    if face_scores:
        active_modalities.append("face")

    if not active_modalities:
        raise ValueError("No modalities provided for fusion.")

    # Normalize weights dynamically
    weights = _normalize_weights(active_modalities)

    # Initialize final scores
    final_scores = {emotion: 0.0 for emotion in ALL_EMOTIONS}

    # Weighted fusion
    for emotion in ALL_EMOTIONS:

        if "text" in active_modalities:
            final_scores[emotion] += (
                text_scores.get(emotion, 0.0) * weights["text"]
            )

        if "voice" in active_modalities:
            final_scores[emotion] += (
                voice_scores.get(emotion, 0.0) * weights["voice"]
            )

        if "face" in active_modalities:
            final_scores[emotion] += (
                face_scores.get(emotion, 0.0) * weights["face"]
            )

    # Determine final emotion
    final_emotion = max(final_scores, key=final_scores.get)
    confidence = round(final_scores[final_emotion], 4)

    logger.info(
        f"Fusion result → {final_emotion} ({confidence}) "
        f"| modalities used: {active_modalities}"
    )

    return {
        "final_emotion": final_emotion,
        "confidence": confidence,
        "fusion_scores": {
            k: round(v, 4) for k, v in final_scores.items()
        },
        "modalities_used": active_modalities,
    }
