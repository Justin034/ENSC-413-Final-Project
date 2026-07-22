
# Real-Time-Audio-Mask
A ML-based audio masking application that removes external noises made with partner Mingyang Cheng

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

# Images and Outcomes
Brief design structure of system\
<img width="739" height="131" alt="Screenshot 2026-07-21 183520" src="https://github.com/user-attachments/assets/824537f1-2a65-4b27-9246-05a0178001c8" />
<img width="621" height="331" alt="Screenshot 2026-07-21 183537" src="https://github.com/user-attachments/assets/c028baec-679e-4d45-bbe5-3080b82c7ffc" />

Outcomes from designs\
<img width="826" height="574" alt="Screenshot 2026-07-21 183625" src="https://github.com/user-attachments/assets/0c374638-bad5-4828-a72f-11b9ad973a6a" />
<img width="806" height="420" alt="Screenshot 2026-07-21 183633" src="https://github.com/user-attachments/assets/25b921a7-8b7f-413c-a23c-ebd2372fc0e2" />

# Notes before proceeding
PESQ python library was unable to be installed using pip, therefore, is not included in the requirements.txt\
Python and Torch version used was the latest versions to accommadate school computers during training, etc.

Folders are as follows:\
test clean => Just speech, no noise\
test input => speech + noise\
test output -> denoised speech
