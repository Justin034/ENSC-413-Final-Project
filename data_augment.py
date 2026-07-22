import os
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

INPUT_DIR = "../train_noise_2"
OUTPUT_DIR = "../train_noise_3"
SAMPLE_RATE = 16000
AUGS_PER_FILE = 5 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# various different audio effect additions
def pitch_shift(y, sr):
    steps = random.uniform(-3, 3)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_stretch(y):
    rate = random.uniform(0.8, 1.25)
    return librosa.effects.time_stretch(y, rate=rate)

def change_volume(y):
    gain = random.uniform(0.5, 1.5)
    return y * gain

# random augment based on chance
def augment(y, sr):
    # Randomly apply augmentations
    if random.random() < 0.7:
        y = pitch_shift(y, sr)
    if random.random() < 0.7:
        y = time_stretch(y)
    if random.random() < 0.5:
        y = change_volume(y)
    
    return y

# augment then save
def process_file(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    target_len = len(y)

    for i in range(AUGS_PER_FILE):
        y_aug = augment(y.copy(), sr)

        out_name = f"{filename}_aug_{i}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        sf.write(out_path, y_aug, sr)

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".wav", ".mp3", ".flac"))]

    for f in tqdm(files):
        process_file(os.path.join(INPUT_DIR, f))

if __name__ == "__main__":
    main()