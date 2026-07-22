import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from model import UNetArch

@torch.inference_mode()
def run_batch_inference(input_folder, output_folder):
    # loading device and data folders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir, output_dir = Path(input_folder), Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # loading model
    model = UNetArch().to(device)
    model.load_state_dict(torch.load("denoiser_model_best.pth", map_location=device, weights_only=True))
    model.eval()

    # STFT params.
    n_fft, hop = 512, 256
    win = torch.hann_window(n_fft).to(device)

    for wav_path in input_dir.glob("*.wav"):
        data, sr = sf.read(wav_path)
        # stereo audio to mono
        if len(data.shape) > 1: 
            data = np.mean(data, axis=1)
        waveform = torch.FloatTensor(data).to(device)

        # data -> STFT and apply model
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop, return_complex=True, window=win, center=True)
        mag = torch.abs(spec).unsqueeze(0).unsqueeze(0)
        mask = model(mag).squeeze()
        
        # send back through ISTFT and save to folder
        clean_spectrogram = torch.polar(torch.abs(spec) * mask, torch.angle(spec))
        clean_waveform = torch.istft(clean_spectrogram, n_fft=n_fft, hop_length=hop, window=win, center=True, length=len(data))
        
        sf.write(output_dir / f"denoised_{wav_path.name}", clean_waveform.cpu().numpy(), sr)
        print(f"Processed: {wav_path.name}")

if __name__ == "__main__": 
    run_batch_inference("test input", "test output")