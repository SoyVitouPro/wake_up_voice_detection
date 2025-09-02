import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

import numpy as np

try:
    import librosa
except ImportError as e:
    raise SystemExit("librosa is required. Please install with: pip install librosa soundfile") from e

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit("PyTorch is required. Please install with: pip install torch") from e


# =============================
# Default configuration (no CLI)
# =============================
# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"

# Audio/feature settings
SR = 16000
DURATION = 1.0
N_MELS = 64

# Training
BATCH_SIZE = 2
EPOCHS = 50
VAL_RATIO = 0.2
SEED = 42

# Export
# For PyTorch we export: state_dict (.pt) and TorchScript (.ts.pt)
# Quantization here refers to dynamic quantization for Linear layers.
QUANTIZE = "dynamic"  # one of: "none", "dynamic"
MODEL_NAME = "wuw_konpov"


def list_audio_files(root: Path, exts=(".wav", ".ogg", ".mp3", ".flac")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def load_audio(
    path: Path,
    sr: int,
    duration: float,
    mono: bool = True,
    offset: Optional[float] = None,
) -> np.ndarray:
    target_len = int(sr * duration)
    y, _ = librosa.load(
        str(path), sr=sr, mono=mono, offset=offset, duration=duration if offset is not None else None
    )
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def compute_logmel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,  # ~10ms at 16kHz
    win_length: int = 400,   # 25ms at 16kHz
) -> np.ndarray:
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


def collect_dataset(
    data_dir: Path,
    sr: int,
    duration: float,
    n_mels: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    random.seed(seed)
    np.random.seed(seed)

    pos_dir = data_dir / "konpov"
    neg_dir = data_dir / "other"

    pos_files = list_audio_files(pos_dir)
    neg_files = list_audio_files(neg_dir)

    if len(pos_files) == 0:
        raise RuntimeError(f"No positive samples found in {pos_dir}")
    if len(neg_files) == 0:
        raise RuntimeError(f"No negative samples found in {neg_dir}")

    X: List[np.ndarray] = []
    y: List[int] = []

    # Balance by downsampling the larger class to the size of the smaller
    min_len = min(len(pos_files), len(neg_files))
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    pos_files = pos_files[:min_len]
    neg_files = neg_files[:min_len]

    all_files = [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]
    print("DAta: ", len(all_files))
    random.shuffle(all_files)

    # Precompute expected time frames for padding
    hop_length = 160
    n_fft = 512
    win_length = 400
    expected_frames = 1 + int(np.floor((sr * duration - n_fft) / hop_length))

    for path, label in all_files:
        wav = load_audio(path, sr=sr, duration=duration)
        feat = compute_logmel(wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # Ensure consistent time dimension by padding/truncating frames
        if feat.shape[1] < expected_frames:
            pad_t = expected_frames - feat.shape[1]
            feat = np.pad(feat, ((0, 0), (0, pad_t)), mode="edge")
        elif feat.shape[1] > expected_frames:
            feat = feat[:, :expected_frames]
        X.append(feat)
        y.append(label)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    meta = {
        "sr": sr,
        "duration": duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "expected_frames": expected_frames,
        "labels": {"konpov": 1, "other": 0},
    }
    return X, y, meta


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * val_ratio)
    if n_val <= 0 and len(X) > 1:
        n_val = 1
    if n_val >= len(X):
        n_val = max(1, len(X) - 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])


def normalize_fit(train_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mean, std, (train_X - mean) / std


def normalize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


class WUWNet(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        def dw_block(in_c, out_c, stride=(1, 1)):
            s_h, s_w = stride
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, stride=(s_h, s_w), padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.block1 = dw_block(16, 24, stride=(2, 2))
        self.block2 = dw_block(24, 32, stride=(2, 2))
        self.block3 = dw_block(32, 48, stride=(1, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(48, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)  # raw logit
        return x


class NumpyAudioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = np.expand_dims(x, axis=0)  # (1, n_mels, frames)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y)


def export_models_torch(model: nn.Module, weights_dir: Path, model_name: str, input_shape: Tuple[int, int, int], quantize: str = "none") -> Tuple[str, str]:
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Save state dict (float)
    pt_path = weights_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), pt_path)

    # TorchScript export (optionally dynamic-quantized on Linear)
    model_export = model.eval().cpu()
    if quantize == "dynamic":
        model_export = torch.quantization.quantize_dynamic(model_export, {nn.Linear}, dtype=torch.qint8)

    example = torch.zeros((1, *input_shape), dtype=torch.float32)
    try:
        ts = torch.jit.trace(model_export, example)
    except Exception:
        ts = torch.jit.script(model_export)
    ts_path = weights_dir / f"{model_name}.ts.pt"
    ts.save(str(ts_path))

    return str(pt_path), str(ts_path)


def save_metadata(weights_dir: Path, model_name: str, meta: dict, norm: dict, input_shape: Tuple[int, int, int]):
    payload = {
        "audio": meta,
        "normalization": norm,
        "model": {
            "name": model_name,
            "input_shape": input_shape,  # [C, H, W] expected by Torch (we pass 1 x n_mels x frames)
            "output": {"activation": "sigmoid", "labels": {"konpov": 1, "other": 0}},
        },
    }
    out = weights_dir / f"{model_name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    data_dir = Path(DATA_DIR)
    weights_dir = Path(WEIGHTS_DIR)

    print(f"Loading dataset from: {data_dir}")
    X, y, meta = collect_dataset(
        data_dir=data_dir,
        sr=SR,
        duration=DURATION,
        n_mels=N_MELS,
        seed=SEED,
    )
    (X_train, y_train), (X_val, y_val) = train_val_split(X, y, val_ratio=VAL_RATIO, seed=SEED)

    mean, std, X_train_n = normalize_fit(X_train)
    X_val_n = normalize_apply(X_val, mean, std)

    # Input to the network is (N, C=1, H=n_mels, W=frames)
    input_shape = (1, X_train_n.shape[1], X_train_n.shape[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WUWNet(in_ch=1).to(device)

    train_ds = NumpyAudioDataset(X_train_n, y_train)
    val_ds = NumpyAudioDataset(X_val_n, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss /= max(1, len(train_ds))
        train_acc = correct / max(1, total)

        model.eval()
        val_loss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                vcorrect += (preds == yb).sum().item()
                vtotal += xb.size(0)
        val_loss /= max(1, len(val_ds))
        val_acc = vcorrect / max(1, vtotal)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    norm = {
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
    }

    pt_path, ts_path = export_models_torch(model, weights_dir, MODEL_NAME, input_shape, quantize=QUANTIZE)
    save_metadata(weights_dir, MODEL_NAME, meta, norm, input_shape)

    print("\nTraining complete.")
    print(f"Saved PyTorch state_dict: {pt_path}")
    print(f"Saved TorchScript model: {ts_path}")
    print(f"Saved metadata JSON: {weights_dir / (MODEL_NAME + '.json')}")
    print("Labels: konpov=1, other=0")


if __name__ == "__main__":
    main()
