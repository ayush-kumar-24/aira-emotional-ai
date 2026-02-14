"""
services/voice_emotion.py
--------------------------
Voice Emotion Detection + Speech-to-Text
"""

import io
import logging
import numpy as np
import tempfile
import os
import uuid

# ====== SSL FIX ======
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
# ====== SSL FIX END ======

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not installed. Voice emotion will return fallback.")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("📥 Loading Whisper model for Speech-to-Text...")
    whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    print("✅ Whisper model loaded successfully!")
except ImportError:
    WHISPER_AVAILABLE = False
    whisper_model = None
    logging.warning("⚠️ Whisper not installed. STT will not work. Run: pip install openai-whisper")
except Exception as e:
    WHISPER_AVAILABLE = False
    whisper_model = None
    logging.error(f"❌ Failed to load Whisper model: {e}")

logger = logging.getLogger(__name__)

TARGET_SR = 22050
EMOTIONS = ["sad", "calm", "neutral", "happy", "excited", "angry", "fearful"]

def _transcribe_audio(audio_bytes: bytes) -> str:
    """Convert speech to text using Whisper (using librosa, no FFmpeg needed)"""
    if not WHISPER_AVAILABLE or whisper_model is None:
        logger.warning("Whisper not available, returning empty transcription")
        return ""
    
    try:
        logger.info("🎤 Transcribing audio...")
        
        # Load audio from bytes using librosa (no FFmpeg dependency!)
        audio_file = io.BytesIO(audio_bytes)
        audio_array, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Transcribe directly from numpy array
        result = whisper_model.transcribe(audio_array, language="en", fp16=False)
        transcription = result["text"].strip()
        
        logger.info(f"📝 Transcription: '{transcription}'")
        return transcription
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def _extract_features(audio_bytes: bytes) -> dict:
    """Extract acoustic features from audio"""
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is not installed. Run: pip install librosa")

    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)

    if len(y) < sr * 0.3:
        raise ValueError("Audio too short for emotion analysis.")

    # Remove silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Ensure we have enough audio after trimming
    if len(y_trimmed) < sr * 0.2:
        y_trimmed = y  # Use original if too much was trimmed
    
    # === ENERGY FEATURES ===
    rms = librosa.feature.rms(y=y_trimmed)[0]
    mean_energy = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_max = float(np.max(rms))
    
    # Dynamic range (ratio of max to mean)
    dynamic_range = energy_max / (mean_energy + 1e-8)
    
    # Coefficient of variation (normalized variance)
    energy_cv = energy_std / (mean_energy + 1e-8)

    # === PITCH FEATURES ===
    f0, voiced_flag, _ = librosa.pyin(
        y_trimmed,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
        sr=sr,
    )

    voiced_f0 = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else np.array([])

    if len(voiced_f0) > 0:
        mean_pitch = float(np.nanmean(voiced_f0))
        pitch_std = float(np.nanstd(voiced_f0))
        pitch_range = float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0))
        # Pitch coefficient of variation
        pitch_cv = pitch_std / (mean_pitch + 1e-8)
    else:
        mean_pitch = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
        pitch_cv = 0.0

    # === SPECTRAL FEATURES ===
    spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    mean_spectral_centroid = float(np.mean(spectral_centroids))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
    mean_spectral_rolloff = float(np.mean(spectral_rolloff))
    
    # === TEMPORAL FEATURES ===
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
    mean_zcr = float(np.mean(zcr))

    # === MFCC (first few coefficients) ===
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    mfcc_mean = [float(np.mean(mfcc)) for mfcc in mfccs]
    
    # === TEMPO ===
    tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
    
    return {
        "mean_energy": mean_energy,
        "energy_std": energy_std,
        "energy_cv": energy_cv,
        "dynamic_range": dynamic_range,
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "pitch_cv": pitch_cv,
        "pitch_range": pitch_range,
        "mean_spectral_centroid": mean_spectral_centroid,
        "mean_spectral_rolloff": mean_spectral_rolloff,
        "mean_zcr": mean_zcr,
        "tempo": float(tempo),
        "mfcc_1": mfcc_mean[0] if len(mfcc_mean) > 0 else 0.0,
        "mfcc_2": mfcc_mean[1] if len(mfcc_mean) > 1 else 0.0,
    }


def _score_emotions(features: dict) -> dict:
    """
    CORRECTED SCORING - Non-overlapping, clear boundaries
    """
    scores = {e: 0.0 for e in EMOTIONS}
    
    # Extract features
    energy_cv = features["energy_cv"]
    pitch_cv = features["pitch_cv"]
    mean_pitch = features["mean_pitch"]
    pitch_range = features["pitch_range"]
    dynamic_range = features["dynamic_range"]
    spectral_centroid = features["mean_spectral_centroid"]
    mean_energy = features["mean_energy"]
    energy_std = features["energy_std"]
    tempo = features["tempo"]
    
    # Log features for debugging
    logger.info(
        f"📊 Features: energy={mean_energy:.4f}, energy_cv={energy_cv:.3f}, "
        f"pitch={mean_pitch:.1f}Hz, pitch_cv={pitch_cv:.3f}, "
        f"pitch_range={pitch_range:.1f}, dynamic_range={dynamic_range:.2f}, "
        f"spectral={spectral_centroid:.1f}, tempo={tempo:.1f}"
    )
    
    # ═══════════════════════════════════════════════════════
    # SAD - Very low energy, monotone, slow
    # ═══════════════════════════════════════════════════════
    sad_score = 0.0
    if mean_energy < 0.015:
        sad_score += 2.5
        logger.info("  ✓ SAD: Very low energy")
    if energy_cv < 0.45:
        sad_score += 2.0
        logger.info("  ✓ SAD: Low energy variation")
    if pitch_cv < 0.12:
        sad_score += 1.5
        logger.info("  ✓ SAD: Monotone pitch")
    if mean_pitch < 140 and mean_pitch > 0:
        sad_score += 1.0
        logger.info("  ✓ SAD: Low pitch")
    scores["sad"] = sad_score
    
    # ═══════════════════════════════════════════════════════
    # CALM - Moderate, steady, controlled
    # ═══════════════════════════════════════════════════════
    calm_score = 0.0
    if 0.012 < mean_energy < 0.025:
        calm_score += 2.0
        logger.info("  ✓ CALM: Moderate low energy")
    if 0.45 < energy_cv < 0.60:
        calm_score += 2.0
        logger.info("  ✓ CALM: Steady energy")
    if 0.10 < pitch_cv < 0.16:
        calm_score += 1.5
        logger.info("  ✓ CALM: Controlled pitch")
    if 2.0 < dynamic_range < 2.9:
        calm_score += 1.0
        logger.info("  ✓ CALM: Controlled dynamics")
    scores["calm"] = calm_score
    
    # ═══════════════════════════════════════════════════════
    # NEUTRAL - Average everything
    # ═══════════════════════════════════════════════════════
    neutral_score = 0.0
    if 0.020 < mean_energy < 0.035:
        neutral_score += 1.5
        logger.info("  ✓ NEUTRAL: Normal energy")
    if 0.50 < energy_cv < 0.70:
        neutral_score += 1.5
        logger.info("  ✓ NEUTRAL: Moderate variation")
    if 140 < mean_pitch < 180:
        neutral_score += 1.0
        logger.info("  ✓ NEUTRAL: Normal pitch")
    if 0.12 < pitch_cv < 0.22:
        neutral_score += 1.0
        logger.info("  ✓ NEUTRAL: Normal pitch variation")
    scores["neutral"] = neutral_score
    
    # ═══════════════════════════════════════════════════════
    # HAPPY - Elevated pitch, brighter, moderate-high energy
    # ═══════════════════════════════════════════════════════
    happy_score = 0.0
    if mean_energy > 0.028:
        happy_score += 2.5
        logger.info("  ✓ HAPPY: High energy")
    if mean_pitch > 170:
        happy_score += 2.5
        logger.info("  ✓ HAPPY: Elevated pitch")
    if pitch_range > 95:
        happy_score += 1.5
        logger.info("  ✓ HAPPY: Wide pitch range")
    if spectral_centroid > 2000:
        happy_score += 1.0
        logger.info("  ✓ HAPPY: Bright timbre")
    scores["happy"] = happy_score
    
    # ═══════════════════════════════════════════════════════
    # EXCITED - Very high energy, dynamic, fast tempo
    # ═══════════════════════════════════════════════════════
    excited_score = 0.0
    if mean_energy > 0.045:
        excited_score += 3.0
        logger.info("  ✓ EXCITED: Very high energy")
    if energy_cv > 0.75:
        excited_score += 2.5
        logger.info("  ✓ EXCITED: Highly dynamic")
    if pitch_range > 125:
        excited_score += 2.0
        logger.info("  ✓ EXCITED: Very wide pitch range")
    if dynamic_range > 3.8:
        excited_score += 1.5
        logger.info("  ✓ EXCITED: High dynamic range")
    if tempo > 140:
        excited_score += 1.0
        logger.info("  ✓ EXCITED: Fast tempo")
    scores["excited"] = excited_score
    
    # ═══════════════════════════════════════════════════════
    # ANGRY - High energy, harsh, tense, lower-mid pitch
    # ═══════════════════════════════════════════════════════
    angry_score = 0.0
    if mean_energy > 0.038:
        angry_score += 2.5
        logger.info("  ✓ ANGRY: High energy")
    if spectral_centroid > 2300:
        angry_score += 2.5
        logger.info("  ✓ ANGRY: Harsh/tense timbre")
    if 130 < mean_pitch < 175:
        angry_score += 1.5
        logger.info("  ✓ ANGRY: Lower-mid pitch")
    if energy_cv > 0.70:
        angry_score += 1.5
        logger.info("  ✓ ANGRY: High energy variation")
    scores["angry"] = angry_score
    
    # ═══════════════════════════════════════════════════════
    # FEARFUL - Unstable, trembling, erratic
    # ═══════════════════════════════════════════════════════
    fearful_score = 0.0
    if pitch_cv > 0.28:
        fearful_score += 3.0
        logger.info("  ✓ FEARFUL: Very unstable pitch")
    if energy_cv > 0.85:
        fearful_score += 2.5
        logger.info("  ✓ FEARFUL: Very unstable energy")
    if pitch_range > 150:
        fearful_score += 2.0
        logger.info("  ✓ FEARFUL: Extreme pitch range")
    scores["fearful"] = fearful_score
    
    # ═══════════════════════════════════════════════════════
    # Normalize scores
    # ═══════════════════════════════════════════════════════
    logger.info(f"📈 Raw scores: {scores}")
    
    max_score = max(scores.values())
    if max_score > 0:
        for k in scores:
            scores[k] = round(scores[k] / max_score, 4)
    else:
        # Fallback to neutral if nothing matched
        scores["neutral"] = 1.0
    
    logger.info(f"📊 Normalized scores: {scores}")
    
    return scores


def analyze_voice(audio_bytes: bytes) -> dict:
    """Main function to analyze voice emotion + transcribe speech"""
    
    # 1. Speech-to-Text
    transcription = _transcribe_audio(audio_bytes)
    
    # 2. Emotion Detection
    features = _extract_features(audio_bytes)
    all_scores = _score_emotions(features)

    emotion = max(all_scores, key=all_scores.get)
    confidence = all_scores[emotion]

    logger.info(
        f"🎯 Final result: {emotion.upper()} (confidence: {confidence:.2f})"
    )

    return {
        "transcription": transcription,  # ⭐ NEW: Speech-to-text
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "features": {k: round(v, 6) for k, v in features.items()},
        "all_scores": all_scores,
    }