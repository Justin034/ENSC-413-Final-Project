import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from model import UNetArch

# adjustable parameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EARLY_STOP_PATIENCE = 7
MIN_DELTA = 1e-4

class Dataset(Dataset):
    # setting up vars. for clean and noisy files
    def __init__(self, clean_dir, noise_dir, sr=16000, duration=3):
        self.clean_files = sorted(list(Path(clean_dir).glob("*.wav")))
        self.noise_files = sorted(list(Path(noise_dir).glob("*.wav")))
        self.sr = sr
        self.target_len = sr * duration
        
        if len(self.clean_files) == 0:
            raise FileNotFoundError(f"No files found.")

    # reading and checking if the file length matches
    def _load_random_chunk(self, path):
        info = sf.info(path)

        # if audio file is too short, it will pad remaining remaining space with 0's
        if info.frames <= self.target_len:
            data, _ = sf.read(path)
            # changing audio file from multiple channels to mono audio if needed
            if len(data.shape) > 1: 
                data = np.mean(data, axis=1)
            data = np.pad(data, (0, self.target_len - len(data)), 'constant')

        # if the audio file is too long, it will select a random chunk of audio of duration target_len
        else:
            max_start = np.random.randint(0, info.frames - self.target_len)
            data, _ = sf.read(path, start=max_start, frames=self.target_len)

            # changing audio file from multiple channels to mono audio if needed
            if len(data.shape) > 1: 
                data = np.mean(data, axis=1)

        return torch.FloatTensor(data)

    def __getitem__(self, idx):
        clean_wf = self._load_random_chunk(self.clean_files[idx])
        noise_wf = self._load_random_chunk(self.noise_files[np.random.randint(0, len(self.noise_files))])

        # mixing the two audio files (scaling noise to produce noise files that are scaled to speech audio loudness)
        c_rms = torch.sqrt(torch.mean(clean_wf**2) + 1e-8)
        n_rms = torch.sqrt(torch.mean(noise_wf**2) + 1e-8)
        snr_factor = (c_rms / (n_rms + 1e-8)) * (10**(-np.random.uniform(-5, 15)/20))
        noisy_wf = clean_wf + (snr_factor * noise_wf)

        # STFT and return
        n_fft, hop = 512, 256
        win = torch.hann_window(n_fft)
        
        c_spec = torch.abs(torch.stft(clean_wf, n_fft, hop, window=win, return_complex=True, center=True))
        n_spec = torch.abs(torch.stft(noisy_wf, n_fft, hop, window=win, return_complex=True, center=True))
        
        return n_spec.unsqueeze(0), c_spec.unsqueeze(0)

    def __len__(self):
        return len(self.clean_files)

# loss function
def scaled_loss(pred, target):

    # L1 loss (applied non-linear factor to scale audio strengths)
    eps = 1e-7 # prevent NaN
    loss = torch.mean(torch.abs(torch.pow(pred + eps, 0.3) - torch.pow(target + eps, 0.3)))

    return loss

def train():
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # loading dataset with 90/10 split
    full_data = Dataset("../data/train/clean", "../train_noise_2", duration=3)
    train_size = int(0.9 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if device.type == 'cuda' else 0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # loading model with AdamW optimizer and learning rate scheduler
    model = UNetArch().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_improve = 0  # early stopping state

    for epoch in range(30):
        model.train()
        train_loss = 0
        
        # model training
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            
            mask = model(noisy)
            loss = scaled_loss(noisy * mask, clean)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        # model validation using new parameters
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                v_mask = model(noisy)
                v_loss = scaled_loss(noisy * v_mask, clean)
                val_loss += v_loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.1e}")

        # saving best model
        if avg_val < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), "denoiser_model_best.pth")
            print("-> Model Improved & Saved.")
        else:
            epochs_no_improve += 1

        # early stopping based on validation loss change
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    train()