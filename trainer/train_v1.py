import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
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


# =============================
# Defaults
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Audio/feature settings
SR = 16000
DURATION = 1.0  # seconds
N_MELS = 64

# Image settings for MobileNetV2
IMG_SIZE = 224  # 224x224 input

# Training settings
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42
NUM_WORKERS = 0


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_audio_files(root: Path, exts=(".wav", ".ogg")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def load_audio_fixed(path: Path, sr: int, duration: float) -> np.ndarray:
    target_len = int(sr * duration)
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
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


class AudioFolderDataset(Dataset):
    """
    Loads audio from data/konpov and data/other, computes log-mel spectrograms,
    and converts them to 3x224x224 tensors for MobileNetV2.
    """

    def __init__(self, data_dir: Path, sr: int, duration: float, n_mels: int, class_map: Dict[str, int]):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.class_map = class_map

        self.items: List[Tuple[Path, int]] = []
        for cname, cid in class_map.items():
            cdir = self.data_dir / cname
            files = list_audio_files(cdir)
            for f in files:
                self.items.append((f, cid))

        if not self.items:
            raise RuntimeError(f"No audio found under {data_dir} with classes {list(class_map.keys())}")

        # Precompute expected frames to pad/crop time dimension consistently
        self._n_fft = 512
        self._hop = 160
        self._win = 400
        expected_frames = 1 + int(np.floor((sr * duration - self._n_fft) / self._hop))
        self.expected_frames = max(expected_frames, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y = load_audio_fixed(path, self.sr, self.duration)
        feat = compute_logmel(y, self.sr, self.n_mels, self._n_fft, self._hop, self._win)

        # Ensure fixed time dimension
        t = feat.shape[1]
        if t < self.expected_frames:
            feat = np.pad(feat, ((0, 0), (0, self.expected_frames - t)), mode="edge")
        elif t > self.expected_frames:
            feat = feat[:, : self.expected_frames]

        # Normalize per-sample (simple z-score)
        mean = feat.mean(axis=(1, ), keepdims=True)
        std = feat.std(axis=(1, ), keepdims=True) + 1e-6
        feat = (feat - mean) / std

        # To torch tensor NCHW (1, H, W)
        x = torch.from_numpy(feat).unsqueeze(0)  # (1, n_mels, frames)
        # Resize to 224x224 and repeat to 3 channels
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        x = x.repeat(3, 1, 1)  # (3, 224, 224)

        return x, torch.tensor(label, dtype=torch.long)


def make_loaders(
    data_dir: Path,
    sr: int,
    duration: float,
    n_mels: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
):
    class_map = {"konpov": 1, "other": 0}
    ds = AudioFolderDataset(data_dir, sr, duration, n_mels, class_map)

    # Stratified split by label indices
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i, (_, lab) in enumerate(ds.items):
        by_class.setdefault(lab, []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for lab, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val_c = int(round(len(idxs) * val_ratio))
        if n_val_c <= 0 and len(idxs) > 1:
            n_val_c = 1
        if n_val_c >= len(idxs):
            n_val_c = max(1, len(idxs) - 1)
        val_idx.extend(idxs[:n_val_c].tolist())
        train_idx.extend(idxs[n_val_c:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Compute class counts in training set for weighting
    train_counts: Dict[int, int] = {c: 0 for c in by_class.keys()}
    for i in train_idx:
        _, lab = ds.items[i]
        train_counts[lab] = train_counts.get(lab, 0) + 1

    return train_loader, val_loader, ds.class_map, train_counts


def build_model(num_classes: int = 2) -> nn.Module:
    # MobileNetV2 from timm without pretrained weights; no extra re-init
    # Common variant: mobilenetv2_100 (1.0 width). Input: 3x224x224.
    m = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=num_classes, in_chans=3)
    return m


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def save_checkpoint(weights_dir: Path, name: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer, best_acc: float, class_map: Dict[str, int]):
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_acc,
        "classes": class_map,
    }
    path = weights_dir / name
    torch.save(ckpt, path)
    return str(path)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = DATA_DIR
    weights_dir = WEIGHTS_DIR

    print(f"Data dir: {data_dir}")
    print(f"Weights dir: {weights_dir}")

    train_loader, val_loader, class_map, train_counts = make_loaders(
        data_dir=data_dir,
        sr=SR,
        duration=DURATION,
        n_mels=N_MELS,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    model = build_model(num_classes=2).to(device)
    # Class-weighted loss to counter imbalance
    # weight[c] = N / (K * n_c)
    num_classes = len(class_map)
    N = sum(train_counts.values()) if len(train_counts) > 0 else 1
    weights = []
    # Ensure order by class index
    for c in range(num_classes):
        n_c = train_counts.get(c, 1)
        w_c = N / (num_classes * max(1, n_c))
        weights.append(w_c)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_path = None
    last_path = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        # Save last checkpoint always (overwrite)
        last_path = save_checkpoint(weights_dir, "model_last.pth", epoch, model, optimizer, best_val_acc, class_map)

        # Save best model based on validation accuracy (overwrite)
        if va_acc >= best_val_acc + 1e-6:
            best_val_acc = va_acc
            best_path = save_checkpoint(weights_dir, "model_best.pth", epoch, model, optimizer, best_val_acc, class_map)

    # Also save a simple label map json for convenience
    labels_json = weights_dir / "labels.json"
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("Training complete.")
    if best_path:
        print(f"Best model: {best_path} (val_acc={best_val_acc:.3f})")
    if last_path:
        print(f"Last checkpoint: {last_path}")
    print(f"Labels saved to: {labels_json}")


if __name__ == "__main__":
    main()
