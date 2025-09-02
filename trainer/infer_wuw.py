import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    import librosa
except ImportError as e:
    raise SystemExit("librosa is required. Install: pip install librosa soundfile") from e


# Defaults (edit if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_NAME = "wuw_konpov"

# Provided test audio path
AUDIO_PATH = "/media/acleda/DATA/code/projects/wake_up_voice_detection/data/other/audio_dataset/tp000001_lika_outdoor.wav"


def load_meta(weights_dir: Path, model_name: str):
    meta_path = weights_dir / f"{model_name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio_fixed(path: Path, sr: int, duration: float) -> np.ndarray:
    target_len = int(sr * duration)
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def compute_logmel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        power=2.0,
        n_mels=n_mels,
        fmin=50,
        fmax=sr // 2,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def prepare_input(audio_path: Path, info: dict) -> Tuple[np.ndarray, np.ndarray]:
    a = info["audio"]
    sr = int(a["sr"])  # 16000
    duration = float(a["duration"])  # 1.0
    n_mels = int(a["n_mels"])  # 64
    n_fft = int(a["n_fft"])  # 512
    hop_length = int(a["hop_length"])  # 160
    win_length = int(a["win_length"])  # 400
    expected_frames = int(a["expected_frames"])  # from training

    y = load_audio_fixed(audio_path, sr=sr, duration=duration)
    feat = compute_logmel(y, sr, n_mels, n_fft, hop_length, win_length)
    # Ensure same time dimension
    if feat.shape[1] < expected_frames:
        feat = np.pad(feat, ((0, 0), (0, expected_frames - feat.shape[1])), mode="edge")
    elif feat.shape[1] > expected_frames:
        feat = feat[:, :expected_frames]

    # Normalize
    norm = info["normalization"]
    mean = np.array(norm["mean"], dtype=np.float32).reshape(-1, 1)
    std = np.array(norm["std"], dtype=np.float32).reshape(-1, 1) + 1e-6
    feat_n = (feat - mean) / std

    # Shape to NCHW: [1, 1, n_mels, frames]
    x = feat_n[np.newaxis, np.newaxis, :, :]
    return feat, x


def main():
    info = load_meta(WEIGHTS_DIR, MODEL_NAME)
    ts_path = WEIGHTS_DIR / f"{MODEL_NAME}.ts.pt"
    if not ts_path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {ts_path}")

    print(f"Loading TorchScript: {ts_path}")
    model = torch.jit.load(str(ts_path), map_location="cpu").eval()

    print(f"Reading audio: {AUDIO_PATH}")
    _, x = prepare_input(AUDIO_PATH, info)

    with torch.no_grad():
        inp = torch.from_numpy(x)
        logits = model(inp)
        prob = torch.sigmoid(logits).item()

    label = 1 if prob >= 0.5 else 0
    label_name = "konpov" if label == 1 else "other"
    print({
        "audio": str(AUDIO_PATH),
        "prob_konpov": float(prob),
        "pred_label": int(label),
        "pred_name": label_name,
    })


if __name__ == "__main__":
    main()

