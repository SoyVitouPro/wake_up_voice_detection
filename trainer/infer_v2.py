import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit("PyTorch is required. Please install with: pip install torch") from e

try:
    from joblib import load as joblib_load
except Exception as e:
    raise SystemExit("joblib is required. Please install with: pip install joblib") from e

try:
    import librosa
except ImportError as e:
    raise SystemExit("librosa is required. Please install with: pip install librosa soundfile") from e


# Defaults (match project structure)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CNN = PROJECT_ROOT / "weights" / "cnn_best.pth"
DEFAULT_SVM = PROJECT_ROOT / "weights" / "svm_best.joblib"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "other" / "testswefw.ogg"

# Audio/feature params (match trainer v2)
SR = 16000
DURATION = 1.0
N_MELS = 64
IMG_SIZE = 224
N_FFT = 512
HOP = 160
WIN = 400


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, emb_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def embed(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


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


def prepare_single_window(y: np.ndarray) -> torch.Tensor:
    feat = compute_logmel(y, SR, N_MELS, N_FFT, HOP, WIN)
    mean = feat.mean(axis=(1,), keepdims=True)
    std = feat.std(axis=(1,), keepdims=True) + 1e-6
    feat = (feat - mean) / std
    x = torch.from_numpy(feat).unsqueeze(0)
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    x = x.repeat(3, 1, 1)
    return x


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


def load_cnn(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    classes: Dict[str, int] = ckpt.get("classes", {"konpov": 1, "other": 0})
    model = SimpleCNN(num_classes=len(classes), emb_dim=128)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, classes, ckpt


def infer(cnn_path: Path, svm_path: Path, audio_path: Path, mode: str = "sliding", hop_s: float = 0.25):
    model, classes, meta = load_cnn(cnn_path)
    inv_classes = {v: k for k, v in classes.items()}
    svm = joblib_load(svm_path)

    if mode == "single":
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        win = int(SR * DURATION)
        if len(y) < win:
            y = np.pad(y, (0, win - len(y)))
        elif len(y) > win:
            y = y[:win]
        x = prepare_single_window(y).unsqueeze(0).float()
        spans = [(0.0, DURATION)]
    else:
        x, spans = prepare_input_sliding(audio_path, window_s=DURATION, hop_s=hop_s)

    with torch.no_grad():
        Z = model.embed(x)  # (N, emb_dim)
    Z_np = Z.cpu().numpy()

    if hasattr(svm, "predict_proba"):
        probs_all = svm.predict_proba(Z_np)  # (N, C)
    else:
        # Fallback using decision_function -> pseudo-probs via softmax
        dec = svm.decision_function(Z_np)
        if dec.ndim == 1:  # binary -> shape (N,)
            dec = np.stack([-dec, dec], axis=1)
        e = np.exp(dec - dec.max(axis=1, keepdims=True))
        probs_all = e / e.sum(axis=1, keepdims=True)

    # Map sklearn's class order to our indices if necessary
    # svm.classes_ contains labels in sorted order used by columns in predict_proba
    # We will reorder to [0, 1, ...] according to our classes dict keys
    svm_order = list(svm.classes_)
    # Build index mapping from our class index -> column in probs_all
    col_idx = [svm_order.index(i) for i in range(len(inv_classes))]
    probs_all = probs_all[:, col_idx]

    konpov_idx = classes.get("konpov", 1)
    best_i = int(np.argmax(probs_all[:, konpov_idx]))
    probs = probs_all[best_i]

    pred_idx = int(np.argmax(probs))
    pred_name = inv_classes.get(pred_idx, str(pred_idx))
    prob_konpov = float(probs[konpov_idx])

    out = {
        "audio": str(audio_path),
        "cnn": str(cnn_path),
        "svm": str(svm_path),
        "mode": mode,
        "window_sec": DURATION,
        "hop_sec": hop_s if mode == "sliding" else 0.0,
        "best_window": {"start": spans[best_i][0], "end": spans[best_i][1]},
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "prob_konpov": prob_konpov,
        "probs": {inv_classes[i]: float(p) for i, p in enumerate(probs)},
        "cnn_epoch": int(meta.get("epoch", -1)),
        "best_val_acc": float(meta.get("best_val_acc", -1.0)),
    }
    print(json.dumps(out, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Infer with CNN embeddings + SVM classifier")
    p.add_argument("--cnn", type=str, default=str(DEFAULT_CNN), help="Path to cnn_best.pth")
    p.add_argument("--svm", type=str, default=str(DEFAULT_SVM), help="Path to svm_best.joblib")
    p.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO), help="Path to input .wav/.ogg audio")
    p.add_argument("--mode", type=str, default="sliding", choices=["single", "sliding"], help="Inference mode")
    p.add_argument("--hop", type=float, default=0.25, help="Hop seconds for sliding mode")
    return p.parse_args()


def main():
    args = parse_args()
    cnn_path = Path(args.cnn)
    svm_path = Path(args.svm)
    audio_path = Path(args.audio)
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN weights not found: {cnn_path}")
    if not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found: {svm_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    infer(cnn_path, svm_path, audio_path, mode=args.mode, hop_s=args.hop)


if __name__ == "__main__":
    main()

