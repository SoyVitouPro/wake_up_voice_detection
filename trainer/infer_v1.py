import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit("PyTorch is required. Please install with: pip install torch") from e

try:
    import timm
except ImportError as e:
    raise SystemExit("timm is required. Please install with: pip install timm") from e

try:
    import librosa
except ImportError as e:
    raise SystemExit("librosa is required. Please install with: pip install librosa soundfile") from e


# Defaults
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "model_best.pth"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "other" / "testswefw.ogg"

# Audio/feature params (match trainer)
SR = 16000
DURATION = 1.0
N_MELS = 64
IMG_SIZE = 224
N_FFT = 512
HOP = 160
WIN = 400


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


def build_model(num_classes: int = 2) -> nn.Module:
    return timm.create_model("mobilenetv2_100", pretrained=False, num_classes=num_classes, in_chans=3)


def prepare_single_window(y: np.ndarray) -> torch.Tensor:
    feat = compute_logmel(y, SR, N_MELS, N_FFT, HOP, WIN)
    mean = feat.mean(axis=(1,), keepdims=True)
    std = feat.std(axis=(1,), keepdims=True) + 1e-6
    feat = (feat - mean) / std
    x = torch.from_numpy(feat).unsqueeze(0)
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    x = x.repeat(3, 1, 1)
    return x  # (3, 224, 224)


def prepare_input_sliding(audio_path: Path, window_s: float = DURATION, hop_s: float = 0.25):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    win = int(SR * window_s)
    hop = max(1, int(SR * hop_s))
    if len(y) <= win:
        y_pad = y
        if len(y_pad) < win:
            y_pad = np.pad(y_pad, (0, win - len(y_pad)))
        x = prepare_single_window(y_pad)
        return x.unsqueeze(0).float(), [(0.0, min(window_s, len(y)/SR))]

    starts = list(range(0, max(1, len(y) - win + 1), hop))
    if starts[-1] + win < len(y):
        starts.append(len(y) - win)
    chunks = []
    spans = []
    for s in starts:
        seg = y[s:s+win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        chunks.append(prepare_single_window(seg))
        spans.append((s / SR, (s + win) / SR))
    x = torch.stack(chunks, dim=0).float()
    return x, spans


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model_state = ckpt.get("model_state", ckpt)
    classes: Dict[str, int] = ckpt.get("classes", {"konpov": 1, "other": 0})
    return model_state, classes, ckpt


def infer(weights_path: Path, audio_path: Path, mode: str = "sliding", hop_s: float = 0.25):
    model_state, classes, meta = load_checkpoint(weights_path)
    inv_classes = {v: k for k, v in classes.items()}

    model = build_model(num_classes=len(inv_classes))
    model.load_state_dict(model_state, strict=True)
    model.eval()

    if mode == "single":
        y = load_audio_fixed(audio_path, SR, DURATION)
        x = prepare_single_window(y).unsqueeze(0).float()
        spans = [(0.0, DURATION)]
    else:
        x, spans = prepare_input_sliding(audio_path, window_s=DURATION, hop_s=hop_s)

    with torch.no_grad():
        logits = model(x)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy()

    konpov_idx = classes.get("konpov", 1)
    konpov_probs = probs_all[:, konpov_idx]
    best_i = int(np.argmax(konpov_probs))
    probs = probs_all[best_i]

    pred_idx = int(np.argmax(probs))
    pred_name = inv_classes.get(pred_idx, str(pred_idx))
    prob_konpov = float(probs[konpov_idx])

    out = {
        "audio": str(audio_path),
        "weights": str(weights_path),
        "mode": mode,
        "window_sec": DURATION,
        "hop_sec": hop_s if mode == "sliding" else 0.0,
        "best_window": {"start": spans[best_i][0], "end": spans[best_i][1]},
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "prob_konpov": prob_konpov,
        "probs": {inv_classes[i]: float(p) for i, p in enumerate(probs)},
        "epoch": int(meta.get("epoch", -1)),
        "best_val_acc": float(meta.get("best_val_acc", -1.0)),
    }
    print(json.dumps(out, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Infer wake-word with timm MobileNetV2")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to model_best.pth")
    p.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO), help="Path to input .wav/.ogg audio")
    p.add_argument("--mode", type=str, default="sliding", choices=["single", "sliding"], help="Inference mode")
    p.add_argument("--hop", type=float, default=0.25, help="Hop seconds for sliding mode")
    return p.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    audio_path = Path(args.audio)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    infer(weights_path, audio_path, mode=args.mode, hop_s=args.hop)


if __name__ == "__main__":
    main()
