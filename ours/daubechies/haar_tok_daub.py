#!/usr/bin/env python3
"""
Daubechies-2 (db2) learned-tokenizer + spatial mixer on ImageNet-100 (single A100, bf16).

Pipeline
  image 3x256x256
    -> Decimated db2 tokenizer (4 dyadic levels, filters initialised to the
       orthonormal db2 bank and learnable, soft orthogonality penalty keeps them
       wavelet-like).  Same decimated layout as haar_tok.py: each level is a
       stride-2 analysis step, the low-pass band feeds the next level, and the
       detail bands of every level are space-to-depth'd onto the token grid.
       db2 filters have 4 taps, so each level uses a 4x4 kernel with periodic
       (circular) padding; the transform of the whole image stays orthonormal,
       but a token's coefficients now draw on pixels just outside its patch.
       Each 16x16 patch becomes a 768-d token (3*16*16 coefficients).
    -> LayerNorm -> Linear -> dim, + position embedding
    -> N blocks: spatial token mixer (depthwise 7x7 conv, or self-attention) + MLP
    -> mean pool -> linear head

The dataset is read from the existing Hugging Face cache (same data root as before).

Usage
    python3 ours/daubechies/haar_tok_daub.py
    python3 ours/daubechies/haar_tok_daub.py --epochs 30 --mixer attn
    python3 ours/daubechies/haar_tok_daub.py --no-learn-filters      # fixed db2 tokenizer (ablation)
    python3 ours/daubechies/haar_tok_daub.py --modulus               # add |detail| features
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[2]   # repository root (shared data/, logs/, runs/)
DATASET_REPO = "ilee0022/ImageNet100"
IMAGE_SIZE = 256
NUM_CLASSES = 100

# Orthonormal Daubechies low-pass analysis filters, in conv2d (cross-correlation) order.
DAUBECHIES_LO = {
    "db2": (0.48296291314469025, 0.836516303737469, 0.22414386804185735, -0.12940952255092145),
    "db4": (0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385,
            -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278),
}


def daubechies_lowpass(n: int) -> tuple[float, ...]:
    """
    Minimum-phase Daubechies low-pass filter with n vanishing moments (2n taps),
    by spectral factorisation; same ordering as DAUBECHIES_LO (reproduces db2/db4 to ~1e-12).
    """
    p = [math.comb(n - 1 + k, k) for k in range(n)]           # P(y), y = sin^2(w/2)
    h = np.array([1.0 + 0j])
    for y in np.roots(p[::-1]):
        r = np.roots([1.0, -(2.0 - 4.0 * y), 1.0])             # (2 - z - 1/z) / 4 = y
        h = np.convolve(h, [1.0, -r[np.argmin(np.abs(r))]])
    for _ in range(n):
        h = np.convolve(h, [1.0, 1.0])
    h = np.real(h)
    return tuple(float(v) for v in h * (math.sqrt(2.0) / h.sum()))


for _n in (6, 8, 12):
    DAUBECHIES_LO[f"db{_n}"] = daubechies_lowpass(_n)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Tee:
    """Duplicate writes to both a file and the original stream."""
    def __init__(self, stream, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = stream
        self._file = open(path, "a", buffering=1)   # line-buffered
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def fileno(self):
        return self._stream.fileno()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_history(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k: (int(v) if k == "epoch" else float(v)) for k, v in r.items()} for r in rows]


# ---------------------------------------------------------------------------
# Shared result files (same format as new.py)
# ---------------------------------------------------------------------------

SHARED_EPOCH_FIELDS = [
    "model", "epoch", "batch_size", "lr", "train_loss", "train_top1",
    "val_loss", "val_top1", "best_top1", "train_images_per_sec",
]
SHARED_FINAL_FIELDS = [
    "model", "status", "best_epoch", "best_top1", "best_val_top1",
    "best_val_loss", "best_train_top1", "best_train_loss",
    "batch_size", "completed_epoch", "epochs",
    "best_checkpoint", "last_checkpoint", "history_csv",
]


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_csv_dump(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _plain(row.get(k, "")) for k in fieldnames})
    os.replace(tmp, path)


def append_shared_epoch_row(runs_root: Path, model_name: str, row: dict, batch_size: int) -> None:
    """Append one epoch row to the shared epoch_metrics.csv."""
    shared_csv = runs_root / "epoch_metrics.csv"
    existing = []
    if shared_csv.is_file():
        with shared_csv.open() as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing
                if not (r.get("model") == model_name and r.get("epoch") == str(row["epoch"]))]
    new_row = {"model": model_name, **row, "batch_size": batch_size,
               "train_images_per_sec": row.get("img_per_sec", "")}
    existing.append(new_row)
    atomic_csv_dump(existing, shared_csv, SHARED_EPOCH_FIELDS)


def update_shared_final(runs_root: Path, model_name: str, summary: dict,
                        best_row: dict, args) -> None:
    """Update shared final_results.csv and final_results.json."""
    run_dir = runs_root / model_name
    final_csv = runs_root / "final_results.csv"
    final_json = runs_root / "final_results.json"
    existing = []
    if final_csv.is_file():
        with final_csv.open() as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("model") != model_name]
    row = {
        "model": model_name,
        "status": "done",
        "best_epoch": best_row.get("epoch", ""),
        "best_top1": summary.get("best_val_top1", ""),
        "best_val_top1": best_row.get("val_top1", ""),
        "best_val_loss": best_row.get("val_loss", ""),
        "best_train_top1": best_row.get("train_top1", ""),
        "best_train_loss": best_row.get("train_loss", ""),
        "batch_size": args.batch_size,
        "completed_epoch": args.epochs,
        "epochs": args.epochs,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "history_csv": str(run_dir / "history.csv"),
    }
    existing.append(row)
    atomic_csv_dump(existing, final_csv, SHARED_FINAL_FIELDS)
    json_rows = [{k: _plain(v) for k, v in r.items()} for r in existing]
    tmp = Path(str(final_json) + ".tmp")
    tmp.write_text(json.dumps(json_rows, indent=2))
    os.replace(tmp, final_json)


# ---------------------------------------------------------------------------
# Data (uses the already-downloaded HF cache)
# ---------------------------------------------------------------------------

class HFImages(Dataset):
    def __init__(self, split, tf):
        self.ds, self.tf = split, tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[int(i)]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(ex["label"])


def load_splits(data_root: Path):
    from datasets import load_dataset

    cache = data_root / "datasets_cache"
    os.environ.setdefault("HF_HOME", str(data_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return load_dataset(DATASET_REPO, cache_dir=str(cache))


def build_loaders(ds, batch_size: int, workers: int, seed: int):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.67, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(round(IMAGE_SIZE / 0.875)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                  prefetch_factor=4 if workers > 0 else None)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(HFImages(ds["train"], train_tf), batch_size=batch_size, shuffle=True,
                       drop_last=True, generator=g, **common)

    def eval_loader(key):
        return DataLoader(HFImages(ds[key], eval_tf), batch_size=batch_size * 2, shuffle=False, **common)

    return train, eval_loader("validation"), (eval_loader("test") if "test" in ds else None)


# ---------------------------------------------------------------------------
# Learned decimated Daubechies-2 tokenizer
# ---------------------------------------------------------------------------

def daubechies_filter_bank(wavelet: str) -> torch.Tensor:
    """Separable 2D bank (LL, LH, HL, HH) as a (4, 1, taps, taps) conv2d weight."""
    lo = torch.tensor(DAUBECHIES_LO[wavelet], dtype=torch.float32)
    signs = torch.tensor([(-1.0) ** n for n in range(len(lo))])
    hi = signs * lo.flip(0)                     # quadrature-mirror high-pass
    return torch.stack((
        torch.outer(lo, lo),    # LL
        torch.outer(lo, hi),    # LH: low-pass vertically, high-pass horizontally
        torch.outer(hi, lo),    # HL
        torch.outer(hi, hi),    # HH
    )).unsqueeze(1)


class DaubechiesTokenizer(nn.Module):
    """
    Multi-level stride-2 analysis, filters initialised to an orthonormal Daubechies
    (LL, LH, HL, HH) bank and optionally trainable (one bank per level, shared
    across colour channels).  Circular padding makes each level an orthonormal
    transform of the whole image at exactly half the resolution, so the token
    layout matches haar_tok.py: detail bands from every level are space-to-depth'd
    onto the final (H/patch x W/patch) grid and each token holds 3 * patch^2 values.
    """

    def __init__(self, patch: int = 16, in_ch: int = 3, learnable: bool = True, modulus: bool = False,
                 wavelet: str = "db2"):
        super().__init__()
        self.levels = int(math.log2(patch))
        if 2 ** self.levels != patch:
            raise ValueError("patch must be a power of two")
        self.patch, self.in_ch, self.modulus = patch, in_ch, modulus
        self.wavelet = wavelet
        self.taps = len(DAUBECHIES_LO[wavelet])
        w = daubechies_filter_bank(wavelet).unsqueeze(0).repeat(self.levels, 1, 1, 1, 1)  # (L,4,1,t,t)
        if learnable:
            self.filters = nn.Parameter(w.clone())
        else:
            self.register_buffer("filters", w.clone())
        base = in_ch * patch * patch
        self.out_channels = base + (base - in_ch if modulus else 0)

    def ortho_penalty(self) -> torch.Tensor:
        """
        Paraunitary penalty per level: the four filters must stay orthonormal to
        each other and to their own copies shifted by every even offset that
        overlaps (0, +-2, ... in each direction).  With 2-tap Haar only the zero
        shift exists, which reduces to haar_tok.py's || W W^T - I ||^2.
        """
        k = self.filters                                   # (L,4,1,t,t)
        L, n = k.shape[0], k.shape[1]
        s = self.taps - 2
        target = torch.zeros(n, n, s + 1, s + 1, device=k.device, dtype=k.dtype)
        centre = s // 2
        target[:, :, centre, centre] = torch.eye(n, device=k.device, dtype=k.dtype)
        total = k.new_zeros(())
        for lvl in range(L):
            bank = k[lvl]                                  # (4,1,t,t)
            padded = F.pad(bank, (s, s, s, s))
            gram = F.conv2d(padded, bank, stride=2)        # (4,4,s+1,s+1): <f_i shifted, f_j>
            total = total + ((gram - target) ** 2).sum()
        return total / L

    def forward(self, x):
        x = x.float()
        B, C, H, W = x.shape
        gh = H // self.patch
        pad = (self.taps - 2) // 2
        cur, details = x, []
        for lvl in range(self.levels):
            w = self.filters[lvl].repeat(C, 1, 1, 1)               # (4C,1,t,t)
            padded = F.pad(cur, (pad, pad, pad, pad), mode="circular")
            y = F.conv2d(padded, w, stride=2, groups=C)            # (B,4C,h,w)
            h, wd = y.shape[-2:]
            y = y.view(B, C, 4, h, wd)
            cur = y[:, :, 0]
            det = y[:, :, 1:].reshape(B, 3 * C, h, wd)
            f = h // gh
            if f > 1:
                det = F.pixel_unshuffle(det, f)                    # -> (B, 3C f^2, gh, gw)
            details.append(det)
        feats = details + [cur]
        if self.modulus:
            feats = feats + [d.abs() for d in details]
        return torch.cat(feats, dim=1)                             # (B, out_channels, gh, gw)


class PixelPatchTokenizer(nn.Module):
    """
    Control for the wavelet tokenizer: each token is the raw pixels of its patch
    (same 3 * patch^2 = 768 values and the same 16x16 grid), so with the shared
    LayerNorm + Linear(768 -> dim) this is the standard ViT learned patch embedding.
    """

    def __init__(self, patch: int = 16, in_ch: int = 3):
        super().__init__()
        self.patch = patch
        self.out_channels = in_ch * patch * patch

    def ortho_penalty(self) -> torch.Tensor:
        return torch.zeros(())

    def forward(self, x):
        return F.pixel_unshuffle(x.float(), self.patch)            # (B, 3 p^2, gh, gw)


# ---------------------------------------------------------------------------
# Spatial-mixing backbone
# ---------------------------------------------------------------------------

def drop_path(x, p: float, training: bool):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
    return x * mask / keep


class ConvMixer(nn.Module):
    def __init__(self, dim, grid, kernel=7):
        super().__init__()
        self.grid = grid
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)

    def forward(self, x):                                          # (B,N,C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.grid, self.grid)
        return self.dw(x).flatten(2).transpose(1, 2)


class AttnMixer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, grid, mixer, kernel, heads, dp, mlp_ratio=4, ls=1e-4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mixer = ConvMixer(dim, grid, kernel) if mixer == "conv" else AttnMixer(dim, heads)
        self.g1 = nn.Parameter(ls * torch.ones(dim))
        self.dp = dp
        self.mlp = None
        if mlp_ratio > 0:
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.g2 = nn.Parameter(ls * torch.ones(dim))
        else:
            self.n2 = None

    def forward(self, x):
        x = x + drop_path(self.g1 * self.mixer(self.n1(x)), self.dp, self.training)
        if self.mlp is not None:
            x = x + drop_path(self.g2 * self.mlp(self.n2(x)), self.dp, self.training)
        return x


class DaubechiesTokNet(nn.Module):
    def __init__(self, patch=16, dim=256, depth=12, mixer="conv", kernel=7, heads=4,
                 drop_path_rate=0.1, learn_filters=True, modulus=False, num_classes=NUM_CLASSES,
                 wavelet="db2", mlp_ratio=4, tokenizer="wavelet"):
        super().__init__()
        grid = IMAGE_SIZE // patch
        if tokenizer == "pixels":
            self.stem = PixelPatchTokenizer(patch, 3)
        else:
            self.stem = DaubechiesTokenizer(patch, 3, learn_filters, modulus, wavelet)
        self.pre = nn.LayerNorm(self.stem.out_channels)
        self.embed = nn.Linear(self.stem.out_channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(dim, grid, mixer, kernel, heads, r, mlp_ratio) for r in rates])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):                # tokenizer in fp32
            t = self.stem(x)
        t = t.flatten(2).transpose(1, 2)                           # (B,N,Ctok)
        z = self.embed(self.pre(t)) + self.pos
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.norm(z).mean(dim=1))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.startswith("stem") or n.endswith("pos") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").double()
        correct += (out.argmax(1) == y).sum()
        n += y.numel()
    return loss_sum.item() / n, correct.item() / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wavelet", choices=list(DAUBECHIES_LO), default="db2")
    p.add_argument("--tokenizer", choices=["wavelet", "pixels"], default="wavelet",
                   help="'pixels' replaces the wavelet with raw-pixel patches (learned patch-embedding control)")
    p.add_argument("--name", default=None, help="run name (default: tok_learn_conv_<wavelet>_100)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--mixer", choices=["conv", "attn"], default="conv")
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4,
                   help="hidden width of the per-token MLP as a multiple of --dim; 0 removes the MLP")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ortho", type=float, default=1e-2, help="orthogonality penalty weight for learnable filters")
    p.add_argument("--no-learn-filters", action="store_true", help="freeze the tokenizer at the exact wavelet")
    p.add_argument("--modulus", action="store_true", help="append |detail| features (scattering-style)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--workers", type=int, default=min(16, max(4, (os.cpu_count() or 8) - 2)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "imagenet100")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "imagenet100")
    args = p.parse_args()
    if args.name is None:
        args.name = "tok_pixels_100" if args.tokenizer == "pixels" else f"tok_learn_conv_{args.wavelet}_100"

    assert torch.cuda.is_available(), "CUDA not available (run with gpurun)"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16
    set_seed(args.seed)

    # Log to both the terminal and logs/<name>.log
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, log_dir / f"{args.name}.log")
    sys.stderr = Tee(sys.__stderr__, log_dir / f"{args.name}.log")

    run_dir = args.runs_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_p, best_p, hist_p = run_dir / "last.pt", run_dir / "best.pt", run_dir / "history.csv"
    (run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    ds = load_splits(args.data_root)
    train_loader, val_loader, test_loader = build_loaders(ds, args.batch_size, args.workers, args.seed)

    model = DaubechiesTokNet(args.patch, args.dim, args.depth, args.mixer, args.kernel, args.heads,
                             args.drop_path, not args.no_learn_filters, args.modulus,
                             wavelet=args.wavelet, mlp_ratio=args.mlp_ratio,
                             tokenizer=args.tokenizer).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, fused=True)
    fwd = torch.compile(model) if args.compile else model

    start_epoch, best, history = 0, 0.0, read_history(hist_p)
    if last_p.is_file():
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["completed_epoch"], ck["best"]
        print(f"resumed from {last_p} at epoch {start_epoch}", flush=True)

    print(f"params={n_params:,} tokens/img={(IMAGE_SIZE // args.patch) ** 2} token_dim={model.stem.out_channels} "
          f"batch={args.batch_size} mixer={args.mixer} wavelet={args.wavelet} learn_filters={not args.no_learn_filters} "
          f"modulus={args.modulus} mlp_ratio={args.mlp_ratio} tokenizer={args.tokenizer}", flush=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    learn = not args.no_learn_filters and args.tokenizer == "wavelet"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_acc = torch.zeros((), device=device, dtype=torch.float64)
        corr_acc = torch.zeros((), device=device, dtype=torch.int64)
        n = 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                out = fwd(x)
            ce = F.cross_entropy(out.float(), y, label_smoothing=args.label_smoothing)
            loss = ce + args.ortho * model.stem.ortho_penalty() if (learn and args.ortho > 0) else ce
            loss.backward()
            opt.step()
            loss_acc += ce.detach().double() * y.numel()
            corr_acc += (out.detach().argmax(1) == y).sum()
            n += y.numel()
            step += 1
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0

        val_loss, val_top1 = evaluate(fwd, val_loader, device, dtype)
        prev_best, best = best, max(best, val_top1)
        row = {
            "epoch": epoch + 1,
            "lr": lr_at(step - 1, total_steps, warmup_steps, args.lr),
            "train_loss": loss_acc.item() / n,
            "train_top1": corr_acc.item() / n,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "best_top1": best,
            "img_per_sec": n / secs,
            "ortho_penalty": float(model.stem.ortho_penalty().detach()) if learn else 0.0,
        }
        history = [r for r in history if r["epoch"] != row["epoch"]] + [row]
        write_history(history, hist_p)
        append_shared_epoch_row(args.runs_root, args.name, row, args.batch_size)
        print(f"[{args.name}] {epoch + 1:03d}/{args.epochs:03d} train_loss={row['train_loss']:.4f} "
              f"train_top1={100 * row['train_top1']:.2f}% val_loss={val_loss:.4f} "
              f"val_top1={100 * val_top1:.2f}% best={100 * best:.2f}% | {row['img_per_sec']:.0f} img/s "
              f"| ortho={row['ortho_penalty']:.2e}", flush=True)

        state = {"completed_epoch": epoch + 1, "best": best, "val_top1": val_top1,
                 "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}
        atomic_save(state, last_p)
        if val_top1 >= prev_best:
            atomic_save(state, best_p)
            print(f"[{args.name}] saved best checkpoint to {best_p}", flush=True)

    # Final: evaluate the best checkpoint once on the untouched test split.
    summary = {"name": args.name, "wavelet": args.wavelet, "params": n_params, "best_val_top1": best}
    if test_loader is not None and best_p.is_file():
        model.load_state_dict(torch.load(best_p, map_location="cpu", weights_only=False)["model"])
        t_loss, t_top1 = evaluate(model, test_loader, device, dtype)
        summary.update({"test_loss": t_loss, "test_top1": t_top1})
        print(f"[{args.name}] TEST top1={100 * t_top1:.2f}% loss={t_loss:.4f}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    best_row = max(history, key=lambda r: r.get("val_top1", 0)) if history else {}
    update_shared_final(args.runs_root, args.name, summary, best_row, args)


if __name__ == "__main__":
    main()
