import torch
import numpy as np
import soundcard as sc
import time
from scipy.io import wavfile
from model import UNetArch

# design params.
FS = 16000
CHUNK = 512 
N_FFT = 512
HOP = 256
BOOST_FACTOR = 20.0

def start_realtime():
    # setting up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetArch().to(device)
    try:
        model.load_state_dict(torch.load("denoiser_model_best.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: 'denoiser_model_best.pth' not found.")
        return
        
    model.eval()


    win = torch.hann_window(N_FFT).to(device)
    mic, speaker = sc.default_microphone(), sc.default_speaker()

    raw_audio_buffer = []
    clean_audio_buffer = []

    try:
        with mic.recorder(samplerate=FS) as recorder, speaker.player(samplerate=FS) as player:
            print(f"Running on {device}. Press Ctrl+C to stop.")
            
            # Constant physical delay based on CHUNK size
            buffer_delay_ms = (CHUNK / FS) * 1000 

            while True:
                # timer 1: Total Loop Start (Includes Soundcard overhead)
                t_loop_start = time.perf_counter()
                
                data = recorder.record(numframes=CHUNK)
                
                # scaling raw input audio
                raw_wf_np = data[:, 0].astype(np.float32) * BOOST_FACTOR
                raw_wf_np = np.clip(raw_wf_np, -1.0, 1.0)
                raw_audio_buffer.append(raw_wf_np)

                # converting to torch 
                wf = torch.from_numpy(raw_wf_np).to(device)
                wf_padded = torch.nn.functional.pad(wf, (N_FFT, N_FFT))

                # Timer 2: Start of STFT to end of iSTFT
                t_transform_start = time.perf_counter()

                # 
                spec = torch.stft(wf_padded, n_fft=N_FFT, hop_length=HOP, 
                                  return_complex=True, window=win, center=True)
                
                mag = torch.abs(spec).unsqueeze(0).unsqueeze(0)
                
                # Timer 3: Pure Model Forward Pass
                t_model_start = time.perf_counter()
                with torch.inference_mode():
                    mask = model(mag).squeeze()
                model_time = (time.perf_counter() - t_model_start) * 1000
                
                clean_spec = torch.polar(torch.abs(spec) * mask, torch.angle(spec))
                
                # ISTFT reconstruction
                clean_wf_padded = torch.istft(clean_spec, n_fft=N_FFT, hop_length=HOP, 
                                              window=win, center=True, length=CHUNK + 2*N_FFT)
                
                transform_time = (time.perf_counter() - t_transform_start) * 1000

                # output
                out_np = clean_wf_padded[N_FFT : N_FFT + CHUNK].cpu().numpy()
                clean_audio_buffer.append(out_np)

                player.play(out_np.reshape(-1, 1))

                # Calculate Total Loop and Combined System Latency
                loop_time = (time.perf_counter() - t_loop_start) * 1000
                total_latency = loop_time + buffer_delay_ms

                # The {:<10} or adding extra spaces ensures the old characters are cleared
                print(f"Model: {model_time:5.2f}ms | Transform/Model: {transform_time:5.2f}ms | Total Loop: {loop_time:5.2f}ms | System Total: {total_latency:5.2f}ms    ", end='\r')

    # when interrupted, saves an audio file with and without audio denoising
    except KeyboardInterrupt:
        print("\nStopping and saving files.")
        
        if len(raw_audio_buffer) > 0:
            final_raw = np.concatenate(raw_audio_buffer)
            final_clean = np.concatenate(clean_audio_buffer)

            wavfile.write("recording_raw.wav", FS, (final_raw * 32767).astype(np.int16))
            wavfile.write("recording_clean.wav", FS, (final_clean * 32767).astype(np.int16))
            
            print(f"Saved 'recording_raw.wav' and 'recording_clean.wav'.")

if __name__ == "__main__":
    start_realtime()