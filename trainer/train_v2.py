import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit("PyTorch is required. Please install with: pip install torch") from e

try:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from joblib import dump as joblib_dump
except Exception as e:
    raise SystemExit(
        "scikit-learn and joblib are required. Please install with: pip install scikit-learn joblib"
    ) from e

try:
    import librosa
except ImportError as e:
    raise SystemExit("librosa is required. Please install with: pip install librosa soundfile") from e


# =============================
# Defaults / Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Audio/feature settings
SR = 16000
DURATION = 1.0
N_MELS = 64
N_FFT = 512
HOP = 160
WIN = 400

# Image settings for CNN
IMG_SIZE = 224

# Training
BATCH_SIZE = 16
EPOCHS = 15
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


class AudioFolderDataset(Dataset):
    def __init__(self, data_dir: Path, sr: int, duration: float, n_mels: int, class_map: Dict[str, int]):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.class_map = class_map

        self.items: List[Tuple[Path, int]] = []
        for cname, cid in class_map.items():
            cdir = self.data_dir / cname
            for f in list_audio_files(cdir):
                self.items.append((f, cid))
        if not self.items:
            raise RuntimeError(f"No audio found under {data_dir} with classes {list(class_map.keys())}")

        expected_frames = 1 + int(np.floor((sr * duration - N_FFT) / HOP))
        self.expected_frames = max(expected_frames, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y = load_audio_fixed(path, self.sr, self.duration)
        feat = compute_logmel(y, self.sr, self.n_mels, N_FFT, HOP, WIN)
        # Fix time dim
        t = feat.shape[1]
        if t < self.expected_frames:
            feat = np.pad(feat, ((0, 0), (0, self.expected_frames - t)), mode="edge")
        elif t > self.expected_frames:
            feat = feat[:, : self.expected_frames]
        # Per-sample z-score across time for each mel
        mean = feat.mean(axis=(1,), keepdims=True)
        std = feat.std(axis=(1,), keepdims=True) + 1e-6
        feat = (feat - mean) / std
        # To 3x224x224
        x = torch.from_numpy(feat).unsqueeze(0)
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        x = x.repeat(3, 1, 1)
        return x, torch.tensor(label, dtype=torch.long)


def stratified_split(ds: AudioFolderDataset, val_ratio: float, seed: int):
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
    return train_idx, val_idx


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


def compute_class_weights(train_idx: List[int], ds: AudioFolderDataset, device) -> torch.Tensor:
    counts: Dict[int, int] = {}
    for i in train_idx:
        _, lab = ds.items[i]
        counts[lab] = counts.get(lab, 0) + 1
    num_classes = len(ds.class_map)
    N = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 1)
        w_c = N / (num_classes * max(1, n_c))
        weights.append(w_c)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return run_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def eval_linear(model, loader, device, criterion):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        run_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return run_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    feats = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        z = model.embed(xb).cpu().numpy()
        feats.append(z)
        labels.append(yb.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def fit_eval_svm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = 42):
    clf = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    return clf, acc


def save_artifacts(weights_dir: Path, tag: str, epoch: int, model: nn.Module, svm: SVC, class_map: Dict[str, int], best_val_acc: float):
    weights_dir.mkdir(parents=True, exist_ok=True)
    # CNN
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_val_acc": best_val_acc,
        "classes": class_map,
        "arch": "SimpleCNN(emb_dim=128)",
    }, weights_dir / f"cnn_{tag}.pth")
    # SVM
    joblib_dump(svm, weights_dir / f"svm_{tag}.joblib")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = DATA_DIR
    weights_dir = WEIGHTS_DIR
    class_map = {"konpov": 1, "other": 0}

    ds = AudioFolderDataset(data_dir, SR, DURATION, N_MELS, class_map)
    train_idx, val_idx = stratified_split(ds, VAL_RATIO, SEED)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SimpleCNN(num_classes=2, emb_dim=128).to(device)
    weight_t, train_counts = compute_class_weights(train_idx, ds, device)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_epoch = -1
    best_svm = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss_lin, va_acc_lin = eval_linear(model, val_loader, device, criterion)

        # SVM evaluation on current embeddings
        X_tr, y_tr = extract_embeddings(model, train_loader, device)
        X_va, y_va = extract_embeddings(model, val_loader, device)
        svm, va_acc_svm = fit_eval_svm(X_tr, y_tr, X_va, y_va, seed=SEED)

        # Save last artifacts
        save_artifacts(weights_dir, "last", epoch, model, svm, class_map, best_val_acc)

        improved = va_acc_svm > best_val_acc + 1e-6
        if improved:
            best_val_acc = va_acc_svm
            best_epoch = epoch
            best_svm = svm
            save_artifacts(weights_dir, "best", epoch, model, svm, class_map, best_val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"cnn_train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"cnn_val_loss={va_loss_lin:.4f} acc={va_acc_lin:.3f} | "
            f"svm_val_acc={va_acc_svm:.3f} | best_svm_acc={best_val_acc:.3f} (epoch {best_epoch})"
        )

    # Save label map
    with open(weights_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print("Training complete.")
    print(f"Best SVM val acc: {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"Saved best CNN: {weights_dir / 'cnn_best.pth'}")
    print(f"Saved best SVM: {weights_dir / 'svm_best.joblib'}")
    print(f"Saved last CNN: {weights_dir / 'cnn_last.pth'}")
    print(f"Saved last SVM: {weights_dir / 'svm_last.joblib'}")


if __name__ == "__main__":
    main()

