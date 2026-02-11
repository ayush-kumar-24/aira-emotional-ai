"""
services/fusion.py
------------------
Multimodal Emotion Fusion Engine
Combines emotions from text, voice, and face
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

EMOTIONS = ["sad", "calm", "neutral", "happy", "excited", "angry", "fearful"]

# Weights for each modality
WEIGHTS = {
    "text": 1.0,
    "voice": 1.2,
    "face": 1.3,
}


def fuse_emotions(
    text_scores: Optional[Dict[str, float]] = None,
    voice_scores: Optional[Dict[str, float]] = None,
    face_scores: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Fuse emotions from multiple modalities with weighted averaging
    
    Args:
        text_scores: Emotion scores from text analysis
        voice_scores: Emotion scores from voice analysis
        face_scores: Emotion scores from face analysis
    
    Returns:
        Dict with fused emotion, confidence, and breakdown
    """
    
    # Check what modalities are available
    available_modalities = []
    if text_scores:
        available_modalities.append("text")
    if voice_scores:
        available_modalities.append("voice")
    if face_scores:
        available_modalities.append("face")
    
    if not available_modalities:
        logger.warning("No emotion scores provided for fusion")
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "modalities_used": [],
            "individual_results": {},
            "fusion_method": "none"
        }
    
    logger.info(f"Fusing emotions from: {', '.join(available_modalities)}")
    
    # CASE 1: Only one modality
    if len(available_modalities) == 1:
        modality = available_modalities[0]
        scores = text_scores if modality == "text" else voice_scores if modality == "voice" else face_scores
        
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]
        
        logger.info(f"Single modality ({modality}): {emotion} ({confidence:.2f})")
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "modalities_used": [modality],
            "individual_results": {
                modality: {"emotion": emotion, "confidence": confidence}
            },
            "fusion_method": "single_modality"
        }
    
    # CASE 2: Multiple modalities - Weighted Fusion
    fused_scores = {e: 0.0 for e in EMOTIONS}
    total_weight = 0.0
    individual_results = {}
    
    # Text
    if text_scores:
        text_emotion = max(text_scores, key=text_scores.get)
        text_confidence = text_scores[text_emotion]
        individual_results["text"] = {
            "emotion": text_emotion,
            "confidence": round(text_confidence, 4),
            "all_scores": text_scores
        }
        
        weight = WEIGHTS["text"]
        for emotion, score in text_scores.items():
            if emotion in fused_scores:
                fused_scores[emotion] += score * weight
        total_weight += weight
        logger.info(f"  Text: {text_emotion} ({text_confidence:.2f})")
    
    # Voice
    if voice_scores:
        voice_emotion = max(voice_scores, key=voice_scores.get)
        voice_confidence = voice_scores[voice_emotion]
        individual_results["voice"] = {
            "emotion": voice_emotion,
            "confidence": round(voice_confidence, 4),
            "all_scores": voice_scores
        }
        
        weight = WEIGHTS["voice"]
        for emotion, score in voice_scores.items():
            if emotion in fused_scores:
                fused_scores[emotion] += score * weight
        total_weight += weight
        logger.info(f"  Voice: {voice_emotion} ({voice_confidence:.2f})")
    
    # Face
    if face_scores:
        face_emotion = max(face_scores, key=face_scores.get)
        face_confidence = face_scores[face_emotion]
        individual_results["face"] = {
            "emotion": face_emotion,
            "confidence": round(face_confidence, 4),
            "all_scores": face_scores
        }
        
        weight = WEIGHTS["face"]
        for emotion, score in face_scores.items():
            if emotion in fused_scores:
                fused_scores[emotion] += score * weight
        total_weight += weight
        logger.info(f"  Face: {face_emotion} ({face_confidence:.2f})")
    
    # Normalize fused scores
    if total_weight > 0:
        for emotion in fused_scores:
            fused_scores[emotion] = round(fused_scores[emotion] / total_weight, 4)
    
    # Get final emotion
    final_emotion = max(fused_scores, key=fused_scores.get)
    final_confidence = fused_scores[final_emotion]
    
    logger.info(f"Fused result: {final_emotion.upper()} ({final_confidence:.2f})")
    
    # Conflict detection
    individual_emotions = [r["emotion"] for r in individual_results.values()]
    conflict_detected = len(set(individual_emotions)) == len(individual_emotions)
    
    if conflict_detected and len(available_modalities) >= 2:
        logger.warning(f"Conflict detected: {individual_emotions}")
    
    return {
        "emotion": final_emotion,
        "confidence": round(final_confidence, 4),
        "modalities_used": available_modalities,
        "individual_results": individual_results,
        "fused_scores": fused_scores,
        "fusion_method": "weighted_average",
        "conflict_detected": conflict_detected
    }


def get_emotion_explanation(emotion: str) -> str:
    """Get a human-readable explanation of the emotion"""
    explanations = {
        "sad": "Feeling down, low energy, withdrawn",
        "calm": "Relaxed, peaceful, composed",
        "neutral": "Balanced, neither positive nor negative",
        "happy": "Joyful, content, positive",
        "excited": "Energetic, enthusiastic, animated",
        "angry": "Frustrated, tense, irritated",
        "fearful": "Anxious, worried, uncertain"
    }
    return explanations.get(emotion, "Unknown emotion")
