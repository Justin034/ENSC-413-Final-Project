# keyboard_noise_live.py
# Base prototype for live keyboard-noise detection + simple suppression

import queue
import threading
import numpy as np
import sounddevice as sd
import librosa
from tensorflow import keras

# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024              # ~64 ms at 16 kHz
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
DETECTION_THRESHOLD = 0.70
ATTENUATION = 0.20             # keep 20% volume when keyboard detected

# Load your trained Keras model
# Expected input shape example: (batch, n_mels, time_frames, 1)
MODEL_PATH = "keyboard_detector.keras"
model = keras.models.load_model(MODEL_PATH)

audio_q = queue.Queue()

def extract_features(audio_chunk: np.ndarray) -> np.ndarray:
    """
    Convert a 1D audio chunk to a normalized mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize roughly to 0..1
    mel_db = (mel_db + 80.0) / 80.0
    mel_db = np.clip(mel_db, 0.0, 1.0)

    # Add channel + batch dimensions
    return mel_db[np.newaxis, ..., np.newaxis].astype(np.float32)

def predict_keyboard_prob(audio_chunk: np.ndarray) -> float:
    """
    Return probability that keyboard noise is present.
    """
    feats = extract_features(audio_chunk)
    pred = model.predict(feats, verbose=0)
    return float(pred[0][0])

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Audio status:", status)

    mono = indata[:, 0].copy().astype(np.float32)

    # Prevent divide-by-zero / unstable features
    if np.max(np.abs(mono)) > 1e-8:
        mono = mono / max(np.max(np.abs(mono)), 1e-8)

    prob = predict_keyboard_prob(mono)

    # Very simple suppression strategy
    if prob >= DETECTION_THRESHOLD:
        processed = mono * ATTENUATION
        print(f"keyboard detected: {prob:.2f}")
    else:
        processed = mono

    outdata[:] = processed.reshape(-1, 1)

def main():
    print("Starting live keyboard-noise detector...")
    print("Press Ctrl+C to stop.")

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()