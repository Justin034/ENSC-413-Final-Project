import os
import numpy as np
import librosa
from tqdm import tqdm
from pystoi import stoi
from pesq import pesq

CLEAN_DIR = "test clean"
DENOISED_DIR = "test output"
OUTPUT_FILE = "evaluation tests/metrics_evaluation_report.txt"
SAMPLE_RATE = 16000

def calculate_metrics(clean, denoised, sr):
    min_len = min(len(clean), len(denoised))
    clean = clean[:min_len]
    denoised = denoised[:min_len]

    # SNR
    clean_pwr = np.sum(clean**2) + 1e-10
    noise_pwr = np.sum((clean - denoised)**2) + 1e-10
    snr = 10 * np.log10(clean_pwr / noise_pwr)

    # STOI
    stoi_score = stoi(clean, denoised, sr, extended=False)

    # PESQ
    try:
        pesq_score = pesq(sr, clean, denoised, 'wb')
    except Exception:
        pesq_score = np.nan

    return snr, stoi_score, pesq_score

def main():
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.lower().endswith((".wav", ".flac"))]
    
    results = []
    
    # file header
    report_lines = [
        "DENOISING EVALUATION REPORT",
        "=" * 70,
        f"{'Filename':<40} | {'SNR (dB)':>8} | {'STOI':>8} | {'PESQ':>8}",
        "-" * 70
    ]

    print(f"Processing {len(clean_files)} files...")

    for filename in tqdm(clean_files):
        clean_path = os.path.join(CLEAN_DIR, filename)
        denoised_filename = f"denoised_{filename}"
        denoised_path = os.path.join(DENOISED_DIR, denoised_filename)

        if not os.path.exists(denoised_path):
            continue

        y_clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE)
        y_denoised, _ = librosa.load(denoised_path, sr=SAMPLE_RATE)

        snr, stoi_val, pesq_val = calculate_metrics(y_clean, y_denoised, SAMPLE_RATE)
        
        results.append((snr, stoi_val, pesq_val))
        
        # add individual file row to report
        pesq_str = f"{pesq_val:.4f}" if not np.isnan(pesq_val) else "Error"
        line = f"{filename[:40]:<40} | {snr:>8.2f} | {stoi_val:>8.4f} | {pesq_str:>8}"
        report_lines.append(line)

    # calculate averages
    if results:
        results_array = np.array(results)
        # filter NaN for PESQ
        avg_snr = np.nanmean(results_array[:, 0])
        avg_stoi = np.nanmean(results_array[:, 1])
        avg_pesq = np.nanmean(results_array[:, 2])

        report_lines.append("-" * 70)
        report_lines.append(f"{'AVERAGE':<40} | {avg_snr:>8.2f} | {avg_stoi:>8.4f} | {avg_pesq:>8.4f}")
        report_lines.append("=" * 70)

    # write to text file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()