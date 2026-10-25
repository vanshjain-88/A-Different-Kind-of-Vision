#!/usr/bin/env python3
"""
Haar learned-tokenizer + spatial mixer on ImageNet-100 (single A100, bf16).

Pipeline
  image 3x256x256
    -> Haar tokenizer (4 dyadic levels, filters initialised to Haar and learnable,
       soft orthogonality penalty keeps them wavelet-like)
       each 16x16 patch becomes a 768-d orthogonal-Haar token (3*16*16 coefficients,
       grouped by scale), on a 16x16 token grid
    -> LayerNorm -> Linear -> dim, + position embedding
    -> N blocks: spatial token mixer (depthwise 7x7 conv, or self-attention) + MLP
    -> mean pool -> linear head

The dataset is read from the existing Hugging Face cache (same data root as before).

Usage
    python3 ours/haar/haar_tok.py
    python3 ours/haar/haar_tok.py --epochs 30 --mixer attn
    python3 ours/haar/haar_tok.py --no-learn-filters      # fixed Haar tokenizer (ablation)
    python3 ours/haar/haar_tok.py --modulus               # add |detail| features
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
# Learned Haar tokenizer
# ---------------------------------------------------------------------------

class HaarTokenizer(nn.Module):
    """
    Multi-level 2x2 stride-2 analysis, filters initialised to the orthonormal Haar
    (LL, LH, HL, HH) and optionally trainable (one 4x4 filter bank per level, shared
    across colour channels). Detail bands from every level are space-to-depth'd onto
    the final (H/patch x W/patch) grid, so each token holds the full multi-scale
    coefficient set of its patch: 3 * patch^2 values. With the fixed Haar init this is
    an exact orthogonal transform of the patch (no information lost).
    """

    def __init__(self, patch: int = 16, in_ch: int = 3, learnable: bool = True, modulus: bool = False):
        super().__init__()
        self.levels = int(math.log2(patch))
        if 2 ** self.levels != patch:
            raise ValueError("patch must be a power of two")
        self.patch, self.in_ch, self.modulus = patch, in_ch, modulus
        haar = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],      # LL
            [[0.5, -0.5], [0.5, -0.5]],    # LH
            [[0.5, 0.5], [-0.5, -0.5]],    # HL
            [[0.5, -0.5], [-0.5, 0.5]],    # HH
        ]).unsqueeze(1)                    # (4,1,2,2)
        w = haar.unsqueeze(0).repeat(self.levels, 1, 1, 1, 1)  # (L,4,1,2,2)
        if learnable:
            self.filters = nn.Parameter(w.clone())
        else:
            self.register_buffer("filters", w.clone())
        base = in_ch * patch * patch
        self.out_channels = base + (base - in_ch if modulus else 0)

    def ortho_penalty(self) -> torch.Tensor:
        """|| W W^T - I ||^2 per level (rows of the 4x4 filter matrix should stay orthonormal)."""
        m = self.filters.reshape(self.levels, 4, 4)
        eye = torch.eye(4, device=m.device, dtype=m.dtype)
        return ((m @ m.transpose(1, 2) - eye) ** 2).sum(dim=(1, 2)).mean()

    def forward(self, x):
        x = x.float()
        B, C, H, W = x.shape
        gh = H // self.patch
        cur, details = x, []
        for lvl in range(self.levels):
            w = self.filters[lvl].repeat(C, 1, 1, 1)               # (4C,1,2,2)
            y = F.conv2d(cur, w, stride=2, groups=C)               # (B,4C,h,w)
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
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.g1 = nn.Parameter(ls * torch.ones(dim))
        self.g2 = nn.Parameter(ls * torch.ones(dim))
        self.dp = dp

    def forward(self, x):
        x = x + drop_path(self.g1 * self.mixer(self.n1(x)), self.dp, self.training)
        x = x + drop_path(self.g2 * self.mlp(self.n2(x)), self.dp, self.training)
        return x


class HaarTokNet(nn.Module):
    def __init__(self, patch=16, dim=256, depth=12, mixer="conv", kernel=7, heads=4,
                 drop_path_rate=0.1, learn_filters=True, modulus=False, num_classes=NUM_CLASSES):
        super().__init__()
        grid = IMAGE_SIZE // patch
        self.stem = HaarTokenizer(patch, 3, learn_filters, modulus)
        self.pre = nn.LayerNorm(self.stem.out_channels)
        self.embed = nn.Linear(self.stem.out_channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(dim, grid, mixer, kernel, heads, r) for r in rates])
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
    p.add_argument("--name", default="haar_tok")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
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
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ortho", type=float, default=1e-2, help="orthogonality penalty weight for learnable filters")
    p.add_argument("--no-learn-filters", action="store_true", help="freeze the tokenizer at exact Haar")
    p.add_argument("--modulus", action="store_true", help="append |detail| features (scattering-style)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--workers", type=int, default=min(16, max(4, (os.cpu_count() or 8) - 2)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "imagenet100")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "imagenet100")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA not available (run with gpurun)"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16
    set_seed(args.seed)

    run_dir = args.runs_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_p, best_p, hist_p = run_dir / "last.pt", run_dir / "best.pt", run_dir / "history.csv"
    (run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    ds = load_splits(args.data_root)
    train_loader, val_loader, test_loader = build_loaders(ds, args.batch_size, args.workers, args.seed)

    model = HaarTokNet(args.patch, args.dim, args.depth, args.mixer, args.kernel, args.heads,
                       args.drop_path, not args.no_learn_filters, args.modulus).to(device)
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
          f"batch={args.batch_size} mixer={args.mixer} learn_filters={not args.no_learn_filters} "
          f"modulus={args.modulus}", flush=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch
    learn = not args.no_learn_filters

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
        print(f"[{args.name}] {epoch + 1:03d}/{args.epochs:03d} train_loss={row['train_loss']:.4f} "
              f"train_top1={100 * row['train_top1']:.2f}% val_loss={val_loss:.4f} "
              f"val_top1={100 * val_top1:.2f}% best={100 * best:.2f}% | {row['img_per_sec']:.0f} img/s "
              f"| ortho={row['ortho_penalty']:.2e}", flush=True)

        state = {"completed_epoch": epoch + 1, "best": best, "val_top1": val_top1,
                 "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}
        atomic_save(state, last_p)
        if val_top1 >= prev_best:
            atomic_save(state, best_p)

    # Final: evaluate the best checkpoint once on the untouched test split.
    summary = {"name": args.name, "params": n_params, "best_val_top1": best}
    if test_loader is not None and best_p.is_file():
        model.load_state_dict(torch.load(best_p, map_location="cpu", weights_only=False)["model"])
        t_loss, t_top1 = evaluate(model, test_loader, device, dtype)
        summary.update({"test_loss": t_loss, "test_top1": t_top1})
        print(f"[{args.name}] TEST top1={100 * t_top1:.2f}% loss={t_loss:.4f}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()