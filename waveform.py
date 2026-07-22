import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# reading files into waveform and confirming sample rate of audio files
file0 = "test clean/clnsp5.wav"
file1 = "test input/clnsp5.wav"
file2 = "test output/denoised_clnsp5.wav"

sr0, data0 = wavfile.read(file0)
sr1, data1 = wavfile.read(file1)
sr2, data2 = wavfile.read(file2)

if sr0 != sr1:
    raise ValueError("Sample rates do not match!")

if sr1 != sr2:
    raise ValueError("Sample rates do not match!")  

# setting up equal data length and calcualting residuals
data0 = data0.astype(np.float32)
data1 = data1.astype(np.float32)
data2 = data2.astype(np.float32)

min_len = min(len(data0), len(data2))
data0 = data0[:min_len]
data1 = data1[:min_len]
data2 = data2[:min_len]

residual = data1 - data2

# setting plot dimensions

time = np.linspace(0, min_len / sr1, num=min_len)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(time, data0)
plt.title("Waveform of Original Audio (Figure 10.1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 2)
plt.plot(time, data1)
plt.title("Waveform of Noisy Audio (Figure 10.2)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(time, data2)
plt.title("Waveform of Denoised Audio (Figure 10.3)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 4)
plt.plot(time, residual)
plt.title("Residual (Noisy - Denoised) (Figure 10.4)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

