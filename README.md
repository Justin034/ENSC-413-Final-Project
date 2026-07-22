# Real-Time-Audio-Mask
A ML-based audio masking application that removes external noises

# System Design

## data_augment.py
As the majority of audio files available online either are only clean or noise files, the data augment file is used to add variations in the audio files. Examples include pitch shifts, time stretch etc.

## inference.py
For testing in non-real time, the inference file is used to apply the masking effect of ML model to determine effectiveness

## metrics.py
Used to determine the audio files final quality based on the following critera: SNR, STOI, PESQ

## model.py
File includes the actual ML model used for the project. Design is roughly based on a U-Net structure but is compressed to accomadate a lower latency rate.

## realtime.py
File used for actually implementing the real-time portion of the project. Similar to the strucutre of the inference file but uses extra features like mic modules to allow for speech.

## waveform.py
Used for determining the final output files and there differences between original and cleaned.

# Notes before proceeding
PESQ python library was unable to be installed using pip, therefore, is not included in the requirements.txt\
Python and Torch version used was the latest versions to accommadate school computers during training, etc.

Folders are as follows:\
test clean => Just speech, no noise\
test input => speech + noise\
test output -> denoised speech
