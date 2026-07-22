import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from preprocess import wav_to_logmel, get_label_from_path


AUDIO_EXTENSIONS = {".wav"}


def find_audio_files(split_dir: Path) -> List[Path]:
    """
    Recursively find audio files under a split directory.
    Example split_dir: dataset/train
    """
    files = []
    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    files.sort()
    return files


def process_split(
    split_dir: Path,
    output_dir: Path,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    target_frames: int = 101,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process one split (train / val / test), save X/y/paths, and return them.
    """
    files = find_audio_files(split_dir)

    if not files:
        raise ValueError(f"No audio files found in {split_dir}")

    X_list = []
    y_list = []
    path_list = []

    print(f"Processing split: {split_dir}")
    print(f"Found {len(files)} audio files")

    for i, file_path in enumerate(files, start=1):
        try:
            x = wav_to_logmel(
                path=str(file_path),
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                target_frames=target_frames,
                normalize=normalize,
            )
            y = get_label_from_path(str(file_path))

            X_list.append(x)
            y_list.append(y)
            path_list.append(str(file_path))

            if i % 100 == 0 or i == len(files):
                print(f"  Processed {i}/{len(files)}")
        except Exception as e:
            print(f"  Skipping {file_path}: {e}")

    if not X_list:
        raise ValueError(f"All files failed in split {split_dir}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    paths = np.array(path_list)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "paths.npy", paths)

    print(f"Saved {output_dir / 'X.npy'} with shape {X.shape}")
    print(f"Saved {output_dir / 'y.npy'} with shape {y.shape}")
    print(f"Saved {output_dir / 'paths.npy'} with shape {paths.shape}")

    return X, y, paths


def main():
    dataset_root = Path("dataset")
    processed_root = Path("processed")

    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = dataset_root / split
        output_dir = processed_root / split

        if not split_dir.exists():
            print(f"Skipping missing split directory: {split_dir}")
            continue

        process_split(
            split_dir=split_dir,
            output_dir=output_dir,
            sr=16000,
            n_mels=64,
            n_fft=400,
            hop_length=160,
            target_frames=101,
            normalize=True,
        )

    print("Done preprocessing dataset.")


if __name__ == "__main__":
    main()