# ENSC-413-Final-Project
Create a live audio filter system using ML 

# System Design

## keyboard_detector_train.py
This script builds and trains a convolutional neural network (CNN) to detect keyboard noise from spectrogram inputs.

## noise_detection.py
This script runs a live audio stream from your microphone, uses a trained model to detect keyboard typing sounds, and reduces the volume whenever those sounds are detected.

## preprocess_dataset.py
This script scans your dataset folders (train/val/test), converts all audio files into log-mel spectrograms, extracts labels, and saves them as .npy files for fast loading during model training.

## preprocess.py
This script provides helper functions for loading audio files, converting them to log-mel spectrograms, normalizing and fixing their size, and extracting labels. It’s used to prepare individual audio files for training or inference.

## train.py
This script loads preprocessed audio data, defines a convolutional neural network for keyboard-click detection, trains it, evaluates on a test set, and saves the final model. It also includes reproducibility, callbacks for early stopping, checkpointing, and learning-rate adjustment.