#!/usr/bin/env python3
"""
Haar operator-mixer ImageNet-100 training for a remote GPU.

Colab / Google Drive / L4-only checks are removed. Core Haar math, models,
augmentations, batch probing, and the Cell 4 training loop are unchanged.

Usage:
    gpurun python3 train.py
    gpurun python3 train.py --models haar_orth haar_over
    gpurun python3 train.py --epochs 100 --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Paths — local project dir, not Colab Drive
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]   # repository root (shared data/, logs/, runs/)
DEFAULT_DATA_ROOT = ROOT / "data" / "imagenet100"
DEFAULT_RUNS_ROOT = ROOT / "runs" / "imagenet100"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "wavevit_compat"

DATASET_REPO = "ilee0022/ImageNet100"
IMAGE_SIZE = 256
NUM_CLASSES = 100
SEED = 42
WAVEVIT_COMMIT = "0b4302e0b10ff12013b03b799dda50393efa69a9"
MODEL_KEYS = ("convnextv2", "deit3", "swinv2", "wavevit", "haar_orth", "haar_over")


def configure_hf_cache(cache_root: Path) -> None:
    hf_home = cache_root / "hf_home"
    ds_cache = cache_root / "datasets_cache"
    hub_cache = hf_home / "hub"
    for p in (hf_home, ds_cache, hub_cache):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ds_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# GPU / AMP (A100: bf16 + TF32; any CUDA GPU is accepted)
# ---------------------------------------------------------------------------

def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run with gpurun so an A100 is visible.")
    return torch.device("cuda:0")


def enable_fast_paths() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "benchmark_limit"):
        torch.backends.cudnn.benchmark_limit = 0
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass


def amp_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


_configure_stdio()


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv_dump(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _plain(row.get(key, "")) for key in fieldnames})
    os.replace(tmp, path)


def atomic_history_dump(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(history).to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def best_history_row(rows: list[dict]) -> Optional[dict]:
    valid = [row for row in rows if "val_top1" in row or "best_top1" in row]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            float(row.get("val_top1", row.get("best_top1", float("-inf")))),
            -int(row.get("epoch", 0)),
        ),
    )


EPOCH_RESULT_FIELDS = (
    "model",
    "epoch",
    "batch_size",
    "lr",
    "train_loss",
    "train_top1",
    "val_loss",
    "val_top1",
    "best_top1",
    "train_images_per_sec",
    "peak_alloc_gb",
    "peak_reserved_gb",
    "vram_total_gb",
)
FINAL_RESULT_FIELDS = (
    "model",
    "status",
    "best_epoch",
    "best_top1",
    "best_val_top1",
    "best_val_loss",
    "best_train_top1",
    "best_train_loss",
    "batch_size",
    "completed_epoch",
    "epochs",
    "best_checkpoint",
    "last_checkpoint",
    "history_csv",
)


def sync_saved_results(runs_root: Path) -> list[dict]:
    """Rewrite the combined result files from every saved model history."""
    epoch_rows = []
    final_rows = []
    for model_key in MODEL_KEYS:
        run_dir = runs_root / model_key
        rows = read_history(run_dir / "history.csv")
        status_path = run_dir / "status.json"
        status = {}
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        if not rows and not status:
            continue
        for row in rows:
            epoch_rows.append({"model": model_key, **row})
        best = best_history_row(rows)
        if best is None:
            best = {
                "epoch": status.get("completed_epoch", ""),
                "batch_size": status.get("batch_size", ""),
                "best_top1": status.get("best_top1", ""),
                "val_top1": status.get("best_top1", ""),
            }
        final_rows.append(
            {
                "model": model_key,
                "status": status.get("status", "unknown"),
                "best_epoch": best.get("epoch", ""),
                "best_top1": best.get("best_top1", best.get("val_top1", "")),
                "best_val_top1": best.get("val_top1", ""),
                "best_val_loss": best.get("val_loss", ""),
                "best_train_top1": best.get("train_top1", ""),
                "best_train_loss": best.get("train_loss", ""),
                "batch_size": best.get("batch_size", status.get("batch_size", "")),
                "completed_epoch": status.get("completed_epoch", best.get("epoch", "")),
                "epochs": status.get("epochs", ""),
                "best_checkpoint": str(run_dir / "best.pt"),
                "last_checkpoint": str(run_dir / "last.pt"),
                "history_csv": str(run_dir / "history.csv"),
            }
        )
        if best_history_row(rows) is not None or status:
            atomic_json_dump(
                {
                    "model": model_key,
                    "status": status.get("status", "unknown"),
                    "best": {key: _plain(value) for key, value in best.items()},
                    "history": [
                        {key: _plain(value) for key, value in row.items()} for row in rows
                    ],
                    "best_checkpoint": str(run_dir / "best.pt"),
                    "last_checkpoint": str(run_dir / "last.pt"),
                },
                run_dir / "results.json",
            )
    atomic_csv_dump(epoch_rows, runs_root / "epoch_metrics.csv", list(EPOCH_RESULT_FIELDS))
    atomic_csv_dump(final_rows, runs_root / "final_results.csv", list(FINAL_RESULT_FIELDS))
    atomic_json_dump(
        [{key: _plain(value) for key, value in row.items()} for row in final_rows],
        runs_root / "final_results.json",
    )
    return final_rows


def format_percent(value) -> str:
    try:
        return f"{100 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def format_loss(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def try_sync_saved_results(runs_root: Path) -> list[dict]:
    try:
        return sync_saved_results(runs_root)
    except Exception:
        traceback.print_exc()
        print(f"[results] failed to update combined results under {runs_root}", flush=True)
        return []


def print_saved_results(runs_root: Path, model_keys) -> None:
    saved = {row["model"]: row for row in try_sync_saved_results(runs_root)}
    print("\nSaved model results", flush=True)
    print(
        f"{'model':<12} {'status':<10} {'epoch':>7} {'best_top1':>10} "
        f"{'val_loss':>10} checkpoint",
        flush=True,
    )
    for model_key in model_keys:
        row = saved.get(model_key)
        if row is None:
            print(f"{model_key:<12} {'missing':<10} {'-':>7} {'-':>10} {'-':>10} -", flush=True)
            continue
        print(
            f"{row['model']:<12} {str(row.get('status', '')):<10} "
            f"{str(row.get('completed_epoch', '')):>7} "
            f"{format_percent(row.get('best_top1')):>10} "
            f"{format_loss(row.get('best_val_loss')):>10} "
            f"{row.get('best_checkpoint', '')}",
            flush=True,
        )
    print(f"Epoch results: {runs_root / 'epoch_metrics.csv'}", flush=True)
    print(f"Final results: {runs_root / 'final_results.csv'}", flush=True)


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def gb(n: int) -> float:
    return n / (1024 ** 3)


# ---------------------------------------------------------------------------
# Wave-ViT vendor (same commit + patches as the notebook)
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise last


def ensure_wavevit_vendor() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    wavevit_py = VENDOR_DIR / "wavevit_compat.py"
    wavelets_py = VENDOR_DIR / "torch_wavelets.py"
    if wavevit_py.is_file() and wavelets_py.is_file():
        return

    raw_base = f"https://raw.githubusercontent.com/YehLi/ImageNetModel/{WAVEVIT_COMMIT}/classification"
    print(f"Fetching Wave-ViT source ({WAVEVIT_COMMIT[:10]}) ...")
    wavevit_src = _fetch_text(raw_base + "/wavevit.py")
    wavelets_src = _fetch_text(raw_base + "/torch_wavelets.py")

    wavevit_src = wavevit_src.replace(
        "from timm.models.layers import DropPath, to_2tuple, trunc_normal_",
        "try:\n    from timm.layers import DropPath, to_2tuple, trunc_normal_\n"
        "except ImportError:\n    from timm.models.layers import DropPath, to_2tuple, trunc_normal_",
    )
    wavevit_src = wavevit_src.replace(
        "from timm.models.registry import register_model",
        "def register_model(fn):\n    return fn",
    )
    wavevit_src = wavevit_src.replace(
        "from timm.models.vision_transformer import _cfg",
        "def _cfg(**kwargs):\n    return {}",
    )
    wavevit_src = wavevit_src.replace("np.int(", "int(")

    for old in [
        "self.filters = self.filters.to(dtype=torch.float16)",
        "self.w_ll = self.w_ll.to(dtype=torch.float16)",
        "self.w_lh = self.w_lh.to(dtype=torch.float16)",
        "self.w_hl = self.w_hl.to(dtype=torch.float16)",
        "self.w_hh = self.w_hh.to(dtype=torch.float16)",
    ]:
        wavelets_src = wavelets_src.replace(old, "")

    wavevit_py.write_text(wavevit_src, encoding="utf-8")
    wavelets_py.write_text(wavelets_src, encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFDatasetAdapter(Dataset):
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(ex["label"])


def prepare_imagenet100(cache_root: Path):
    from datasets import DatasetDict, load_dataset

    ds_cache = os.environ.get("HF_DATASETS_CACHE", str(cache_root / "datasets_cache"))
    ds = load_dataset(DATASET_REPO, cache_dir=ds_cache)
    if not isinstance(ds, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict from {DATASET_REPO}, got {type(ds)}")
    if "train" not in ds:
        raise RuntimeError(f"Dataset has no train split. Found: {list(ds.keys())}")
    eval_key = next((k for k in ("validation", "val", "test") if k in ds), None)
    if eval_key is None:
        raise RuntimeError(f"No validation/val/test split. Found: {list(ds.keys())}")

    train_labels = ds["train"].features.get("label")
    if train_labels is not None and hasattr(train_labels, "num_classes"):
        ncls = int(train_labels.num_classes)
    else:
        ncls = len(set(map(int, ds["train"]["label"])))
    if ncls != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, dataset exposes {ncls}")

    meta = {
        "repo": DATASET_REPO,
        "splits": {k: len(v) for k, v in ds.items()},
        "train_split": "train",
        "eval_split": eval_key,
        "num_classes": ncls,
        "cache_dir": ds_cache,
    }
    atomic_json_dump(meta, DEFAULT_RUNS_ROOT / "dataset_manifest.json")
    return ds, eval_key


def build_transforms(image_size: int = IMAGE_SIZE):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.67, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
    ])
    val_resize = int(round(image_size / 0.875))
    val_tf = transforms.Compose([
        transforms.Resize(val_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def worker_count() -> int:
    c = os.cpu_count() or 8
    return min(12, max(4, c - 2))


def build_loaders(ds, eval_key: str, batch_size: int, seed: int = SEED):
    train_tf, val_tf = build_transforms(IMAGE_SIZE)
    train_ds = HFDatasetAdapter(ds["train"], train_tf)
    val_ds = HFDatasetAdapter(ds[eval_key], val_tf)
    nw = worker_count()
    common = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Orthogonal Haar operator and inverse
# ---------------------------------------------------------------------------

def haar2d_level(x):
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def ihaar2d_level(ll, lh, hl, hh):
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    *lead, h, w = ll.shape
    out = torch.empty(*lead, h * 2, w * 2, dtype=ll.dtype, device=ll.device)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def orthogonal_haar_projector_bank(x, levels):
    """Return [B, 1+3J, C, H, W] orthogonal Haar subspace projections P_r x."""
    cur = x
    details = []
    for _ in range(levels):
        cur, lh, hl, hh = haar2d_level(cur)
        details.append((lh, hl, hh))

    components = []
    z = cur
    for _ in range(levels, 0, -1):
        q = torch.zeros_like(z)
        z = ihaar2d_level(z, q, q, q)
    components.append(z)

    for j in range(1, levels + 1):
        ref = details[j - 1][0]
        for band_idx in range(3):
            q = torch.zeros_like(ref)
            bands = [q, q, q]
            bands[band_idx] = details[j - 1][band_idx]
            z = ihaar2d_level(q, bands[0], bands[1], bands[2])
            for _ in range(j - 1, 0, -1):
                q2 = torch.zeros_like(z)
                z = ihaar2d_level(z, q2, q2, q2)
            components.append(z)

    return torch.stack(components, dim=1)


# ---------------------------------------------------------------------------
# Overcomplete tight Haar frame (undecimated) and adjoints
# h=[1,1]/2, g=[1,-1]/2 => Parseval analysis at each level without decimation.
# ---------------------------------------------------------------------------

def tight_haar_frame_level(x, dilation):
    sx = torch.roll(x, shifts=-dilation, dims=-1)
    lx = (x + sx) * 0.5
    hx = (x - sx) * 0.5
    sy_lx = torch.roll(lx, shifts=-dilation, dims=-2)
    sy_hx = torch.roll(hx, shifts=-dilation, dims=-2)
    ll = (lx + sy_lx) * 0.5
    hl = (lx - sy_lx) * 0.5
    lh = (hx + sy_hx) * 0.5
    hh = (hx - sy_hx) * 0.5
    return ll, lh, hl, hh


def tight_haar_frame_adjoint_band(y, dilation, band):
    if band in ("ll", "lh"):
        tmp = (y + torch.roll(y, shifts=dilation, dims=-2)) * 0.5
    else:
        tmp = (y - torch.roll(y, shifts=dilation, dims=-2)) * 0.5
    if band in ("ll", "hl"):
        out = (tmp + torch.roll(tmp, shifts=dilation, dims=-1)) * 0.5
    else:
        out = (tmp - torch.roll(tmp, shifts=dilation, dims=-1)) * 0.5
    return out


def overcomplete_haar_projector_bank(x, levels):
    """Return [B, 1+3J, C, H, W] synthesis components Psi_r^* Psi_r x of a tight frame."""
    cur = x
    details = []
    for j in range(levels):
        d = 2 ** j
        cur, lh, hl, hh = tight_haar_frame_level(cur, d)
        details.append((lh, hl, hh))

    components = []
    z = cur
    for j in reversed(range(levels)):
        z = tight_haar_frame_adjoint_band(z, 2 ** j, "ll")
    components.append(z)

    for j in range(levels):
        d = 2 ** j
        for band_name, coeff in zip(("lh", "hl", "hh"), details[j]):
            z = tight_haar_frame_adjoint_band(coeff, d, band_name)
            for k in reversed(range(j)):
                z = tight_haar_frame_adjoint_band(z, 2 ** k, "ll")
            components.append(z)

    return torch.stack(components, dim=1)


# ---------------------------------------------------------------------------
# Dense operator mixer — no QKV, no learned embedding, no sparsity
# ---------------------------------------------------------------------------

class DenseOperatorMixerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mix = nn.Parameter(torch.empty(dim, dim))
        nn.init.normal_(self.mix, mean=0.0, std=0.02)

    def forward(self, z):
        rms = torch.rsqrt(z.square().mean(dim=1, keepdim=True) + 1e-6)
        zn = z * rms
        mixed = torch.einsum("oi,bihw->bohw", self.mix, zn)
        return z + F.gelu(mixed)


class HaarOperatorMixer(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, depth=12, overcomplete=False):
        super().__init__()
        if image_size <= 0 or (image_size & (image_size - 1)) != 0:
            raise ValueError("image_size must be a positive power of two for the full dyadic Haar depth used here")
        self.image_size = image_size
        self.levels = int(math.log2(image_size))
        self.K = 1 + 3 * self.levels
        self.C = 3
        self.D = self.K * self.C
        self.overcomplete = bool(overcomplete)
        self.blocks = nn.ModuleList([DenseOperatorMixerBlock(self.D) for _ in range(depth)])
        self.head = nn.Linear(self.D, num_classes)

    def operator_bank(self, x):
        if self.overcomplete:
            return overcomplete_haar_projector_bank(x, self.levels)
        return orthogonal_haar_projector_bank(x, self.levels)

    def forward(self, x):
        z = self.operator_bank(x)
        b, k, c, h, w = z.shape
        z = z.reshape(b, k * c, h, w)
        for block in self.blocks:
            z = block(z)
        z = z.mean(dim=(-2, -1))
        return self.head(z)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _choose_timm_name(exact, fallback_substrings):
    import timm

    names = timm.list_models(pretrained=False)
    if exact in names:
        return exact
    matches = [n for n in names if all(s in n for s in fallback_substrings)]
    if not matches:
        raise RuntimeError(f"No compatible timm model found for {exact}; timm={timm.__version__}")
    return sorted(matches)[0]


def resolved_baseline_names():
    return {
        "convnextv2": _choose_timm_name("convnextv2_tiny.fcmae_ft_in1k", ("convnextv2", "tiny")),
        "deit3": _choose_timm_name("deit3_small_patch16_224.fb_in1k", ("deit3", "small", "patch16")),
        "swinv2": _choose_timm_name("swinv2_tiny_window8_256.ms_in1k", ("swinv2", "tiny", "256")),
    }


def build_wavevit(num_classes=NUM_CLASSES):
    ensure_wavevit_vendor()
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        import wavevit_compat
    except Exception as e:
        raise RuntimeError("Wave-ViT compatibility source is missing/broken.") from e
    return wavevit_compat.wavevit_s(
        pretrained=False, num_classes=num_classes, token_label=False, drop_path_rate=0.1
    )


def build_model(model_key: str):
    import timm

    if model_key == "haar_orth":
        return HaarOperatorMixer(overcomplete=False)
    if model_key == "haar_over":
        return HaarOperatorMixer(overcomplete=True)
    if model_key == "wavevit":
        return build_wavevit()

    names = resolved_baseline_names()
    if model_key == "convnextv2":
        return timm.create_model(names[model_key], pretrained=False, num_classes=NUM_CLASSES, drop_path_rate=0.1)
    if model_key == "deit3":
        return timm.create_model(
            names[model_key], pretrained=False, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE, drop_path_rate=0.1
        )
    if model_key == "swinv2":
        return timm.create_model(
            names[model_key], pretrained=False, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE, drop_path_rate=0.1
        )
    raise ValueError(f"Unknown model key: {model_key}")


# ---------------------------------------------------------------------------
# Training helpers (Cell 4)
# ---------------------------------------------------------------------------

def logits_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x) and x.ndim == 2:
                return x
    if isinstance(output, dict):
        for key in ("logits", "out", "pred", "prediction"):
            x = output.get(key)
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Unsupported model output type: {type(output)}")


def to_channels_last(model):
    try:
        return model.to(memory_format=torch.channels_last)
    except Exception:
        return model


def make_adamw(model, lr, weight_decay):
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


def make_scaler(dtype: torch.dtype):
    if dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def step_optimizer(loss, optimizer, scaler):
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def epoch_lr(epoch: int, base_lr: float, warmup: int, total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    denominator = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / float(denominator)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def clean_state_dict(state_dict):
    clean = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        clean[key] = value
    return clean


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_resume(model_key, model, optimizer, last_path: Path, legacy_path: Path):
    candidates = [p for p in (last_path, legacy_path) if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            checkpoint = torch_load(path)
            state_dict = checkpoint.get(
                "model", checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
            )
            if not isinstance(state_dict, dict):
                continue
            result = model.load_state_dict(clean_state_dict(state_dict), strict=False)
            if len(result.missing_keys) > max(5, len(model.state_dict()) // 5):
                continue
            optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
            if isinstance(optimizer_state, dict):
                try:
                    optimizer.load_state_dict(optimizer_state)
                except Exception as e:
                    print(f"[{model_key}] optimizer state not reused: {e}")
            if "completed_epoch" in checkpoint:
                start_epoch = int(checkpoint["completed_epoch"])
            else:
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best = float(checkpoint.get("best_acc", checkpoint.get("best_top1", checkpoint.get("best_val_acc", 0.0))))
            print(f"[{model_key}] resumed from {path} at epoch {start_epoch}")
            return start_epoch, best
        except Exception as e:
            print(f"[{model_key}] ignoring incompatible checkpoint {path}: {e}")
    return 0, 0.0


# ---------------------------------------------------------------------------
# Adaptive batch probe (same algorithm as Cell 4; higher cap on 80GB)
# ---------------------------------------------------------------------------

@dataclass
class TrainSettings:
    epochs: int = 100
    warmup_epochs: int = 5
    base_lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    seed: int = SEED
    batch_cap: int = 4096
    probe_anchor: int = 128


def probe_one(model_key: str, batch_size: int, settings: TrainSettings, device, dtype):
    model = optimizer = scaler = x = y = None
    try:
        cuda_cleanup()
        torch.cuda.reset_peak_memory_stats()
        model = to_channels_last(build_model(model_key).to(device))
        model.train()
        optimizer = make_adamw(model, settings.base_lr, settings.weight_decay)
        scaler = make_scaler(dtype)
        x = torch.empty(
            (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
            device=device,
            dtype=torch.float32,
            memory_format=torch.channels_last,
        ).normal_()
        y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                out = logits_from_output(model(x))
                loss = F.cross_entropy(out, y)
            step_optimizer(loss, optimizer, scaler)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "fit": True,
            "batch_size": int(batch_size),
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
            "images_per_sec": 2.0 * batch_size / max(elapsed, 1e-9),
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            return {"fit": False, "batch_size": int(batch_size), "error": "OOM"}
        raise
    finally:
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        del model, optimizer, scaler, x, y
        cuda_cleanup()


def find_max_batch(model_key: str, settings: TrainSettings, runs_root: Path, device, dtype, gpu_name: str) -> int:
    run_dir = runs_root / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "batch_probe.json"
    signature = {
        "probe_version": 4,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "image_size": IMAGE_SIZE,
        "amp_dtype": str(dtype),
    }
    if cache_path.is_file():
        try:
            old = json.loads(cache_path.read_text(encoding="utf-8"))
            comparable = (
                old.get("gpu") == gpu_name
                and old.get("torch") == torch.__version__
                and int(old.get("image_size", -1)) == IMAGE_SIZE
                and old.get("amp_dtype") == str(dtype)
            )
            if comparable:
                bs = int(old["selected_batch"])
                print(f"[{model_key}] reusing measured batch = {bs}")
                return bs
        except Exception:
            pass

    tested = {}

    def test(bs: int):
        bs = int(bs)
        if bs not in tested:
            print(f"[batch probe:{model_key}] {bs} ... ", end="", flush=True)
            tested[bs] = probe_one(model_key, bs, settings, device, dtype)
            result = tested[bs]
            if result["fit"]:
                print(
                    f"FIT | alloc={result['peak_alloc_gb']:.2f} GB "
                    f"reserved={result['peak_reserved_gb']:.2f} GB | "
                    f"{result['images_per_sec']:.1f} img/s"
                )
            else:
                print("OOM")
        return tested[bs]

    cap = settings.batch_cap
    anchor = min(settings.probe_anchor, cap)
    result = test(anchor)
    if result["fit"]:
        low = anchor
        high = None
        candidate = anchor * 2
        while candidate <= cap:
            if test(candidate)["fit"]:
                low = candidate
                candidate *= 2
            else:
                high = candidate
                break
        if high is None:
            high = low + 256
            while high <= cap and test(high)["fit"]:
                low = high
                high += 256
            if high > cap:
                high = None
    else:
        high = anchor
        low = None
        candidate = anchor // 2
        while candidate >= 8:
            if test(candidate)["fit"]:
                low = candidate
                break
            high = candidate
            candidate //= 2
        if low is None:
            raise RuntimeError(f"{model_key}: batch size 8 does not fit.")

    if high is not None:
        while (high - low) > 8:
            mid = ((low + high) // 16) * 8
            mid = max(low + 8, min(high - 8, mid))
            if test(mid)["fit"]:
                low = mid
            else:
                high = mid

    selected = int(low)
    record = dict(signature)
    record["selected_batch"] = selected
    record["tested"] = [tested[k] for k in sorted(tested)]
    atomic_json_dump(record, cache_path)
    print(f"[{model_key}] maximum measured training batch = {selected}")
    return selected


# ---------------------------------------------------------------------------
# Train one model (Cell 4 loop)
# ---------------------------------------------------------------------------

def train_one(
    model_key: str,
    ds,
    eval_key: str,
    settings: TrainSettings,
    runs_root: Path,
    device,
    dtype,
    gpu_name: str,
    fixed_batch: Optional[int],
) -> None:
    run_dir = runs_root / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    legacy_last = run_dir / "l4_last.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    status_path = run_dir / "status.json"

    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "done" and int(status.get("completed_epoch", 0)) >= settings.epochs:
                print(
                    f"[{model_key}] already completed {settings.epochs} epochs; skipping.",
                    flush=True,
                )
                saved = {
                    row["model"]: row for row in try_sync_saved_results(runs_root)
                }.get(model_key)
                if saved is not None:
                    print(
                        f"[{model_key}] best={format_percent(saved.get('best_top1'))} "
                        f"val_loss={format_loss(saved.get('best_val_loss'))} "
                        f"best_checkpoint={saved.get('best_checkpoint')} "
                        f"history={saved.get('history_csv')}",
                        flush=True,
                    )
                return
        except Exception:
            pass

    batch_size = int(fixed_batch) if fixed_batch else find_max_batch(
        model_key, settings, runs_root, device, dtype, gpu_name
    )
    train_loader, val_loader = build_loaders(ds, eval_key, batch_size, seed=settings.seed)

    cuda_cleanup()
    model = to_channels_last(build_model(model_key).to(device))
    optimizer = make_adamw(model, settings.base_lr, settings.weight_decay)
    scaler = make_scaler(dtype)
    start_epoch, best_top1 = load_resume(model_key, model, optimizer, last_path, legacy_last)
    model = model.to(device)

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=settings.label_smoothing)
    val_loss_fn = nn.CrossEntropyLoss()

    history = []
    if history_path.is_file():
        try:
            history = pd.read_csv(history_path).to_dict("records")
        except Exception:
            history = []

    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(
        f"[{model_key}] params={count_trainable(model):,}  batch={batch_size}  "
        f"start_epoch={start_epoch}  best={100 * best_top1:.2f}%",
        flush=True,
    )
    print(
        f"[{model_key}] checkpoints: last={last_path} best={best_path} history={history_path}",
        flush=True,
    )

    for epoch in range(start_epoch, settings.epochs):
        lr = epoch_lr(epoch, settings.base_lr, settings.warmup_epochs, settings.epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        torch.cuda.reset_peak_memory_stats()
        train_loss_acc = torch.zeros((), device=device, dtype=torch.float64)
        train_correct_acc = torch.zeros((), device=device, dtype=torch.int64)
        train_n = 0
        epoch_start = time.perf_counter()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                out = logits_from_output(model(images))
                loss = train_loss_fn(out, targets)
            step_optimizer(loss, optimizer, scaler)
            n = targets.numel()
            train_loss_acc.add_(loss.detach().double() * n)
            train_correct_acc.add_((out.detach().argmax(1) == targets).sum())
            train_n += n

        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        train_loss = train_loss_acc.item() / max(1, train_n)
        train_top1 = train_correct_acc.item() / max(1, train_n)

        model.eval()
        val_loss_acc = torch.zeros((), device=device, dtype=torch.float64)
        val_correct_acc = torch.zeros((), device=device, dtype=torch.int64)
        val_n = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=dtype, enabled=True):
                    out = logits_from_output(model(images))
                    vloss = val_loss_fn(out, targets)
                n = targets.numel()
                val_loss_acc.add_(vloss.detach().double() * n)
                val_correct_acc.add_((out.argmax(1) == targets).sum())
                val_n += n

        val_loss = val_loss_acc.item() / max(1, val_n)
        val_top1 = val_correct_acc.item() / max(1, val_n)
        previous_best = best_top1
        best_top1 = max(best_top1, val_top1)

        row = {
            "epoch": epoch + 1,
            "batch_size": batch_size,
            "lr": lr,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "best_top1": best_top1,
            "train_images_per_sec": train_n / max(train_seconds, 1e-9),
            "peak_alloc_gb": gb(peak_alloc),
            "peak_reserved_gb": gb(peak_reserved),
            "vram_total_gb": gb(total_vram),
        }
        history = [r for r in history if int(r.get("epoch", -1)) != epoch + 1]
        history.append(row)
        history.sort(key=lambda r: int(r["epoch"]))
        atomic_history_dump(history, history_path)
        print(
            f"[{model_key}] {epoch + 1:03d}/{settings.epochs:03d} "
            f"bs={batch_size} train_loss={train_loss:.4f} train_top1={100 * train_top1:.2f}% "
            f"val_loss={val_loss:.4f} val_top1={100 * val_top1:.2f}% best={100 * best_top1:.2f}% "
            f"| {row['train_images_per_sec']:.1f} img/s "
            f"| VRAM alloc/res={row['peak_alloc_gb']:.2f}/{row['peak_reserved_gb']:.2f} GB",
            flush=True,
        )

        state = {
            "model_key": model_key,
            "epoch": epoch,
            "completed_epoch": epoch + 1,
            "best_acc": best_top1,
            "val_top1": val_top1,
            "val_loss": val_loss,
            "train_top1": train_top1,
            "train_loss": train_loss,
            "batch_size": batch_size,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "amp_dtype": str(dtype),
        }
        atomic_torch_save(state, last_path)
        print(f"[{model_key}] saved checkpoint to {last_path}", flush=True)
        if val_top1 >= previous_best:
            atomic_torch_save(state, best_path)
            print(f"[{model_key}] saved best checkpoint to {best_path}", flush=True)

        atomic_json_dump(
            {
                "status": "running",
                "model": model_key,
                "completed_epoch": epoch + 1,
                "epochs": settings.epochs,
                "batch_size": batch_size,
                "best_top1": best_top1,
                "latest": {key: _plain(value) for key, value in row.items()},
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(last_path),
                "history_csv": str(history_path),
            },
            status_path,
        )
        try_sync_saved_results(runs_root)

    latest = history[-1] if history else {}
    atomic_json_dump(
        {
            "status": "done",
            "model": model_key,
            "completed_epoch": settings.epochs,
            "epochs": settings.epochs,
            "batch_size": batch_size,
            "best_top1": best_top1,
            "latest": {key: _plain(value) for key, value in latest.items()},
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history_csv": str(history_path),
        },
        status_path,
    )
    try_sync_saved_results(runs_root)
    print(
        f"[{model_key}] finished epochs={settings.epochs} "
        f"best={format_percent(best_top1)} best_checkpoint={best_path}",
        flush=True,
    )
    del model, optimizer, scaler, train_loader, val_loader
    cuda_cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Haar operator-mixer ImageNet-100 training")
    p.add_argument("--models", nargs="+", default=list(MODEL_KEYS), choices=list(MODEL_KEYS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch-size", type=int, default=0, help="Fixed batch size. 0 = auto-probe max that fits.")
    p.add_argument("--batch-cap", type=int, default=4096, help="Upper bound for auto batch probe (A100 80GB).")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return p.parse_args()


def main():
    args = parse_args()
    configure_hf_cache(args.data_root)
    device = require_cuda()
    enable_fast_paths()
    set_seed(args.seed)

    gpu_name = torch.cuda.get_device_name(0)
    dtype = amp_dtype()
    vram = torch.cuda.get_device_properties(0).total_memory

    args.runs_root.mkdir(parents=True, exist_ok=True)
    ensure_wavevit_vendor()
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))

    print(
        f"GPU: {gpu_name} | VRAM: {gb(vram):.2f} GB | AMP: {dtype} | "
        f"torch {torch.__version__}"
    )
    print(f"Data cache: {args.data_root}")
    print(f"Checkpoints: {args.runs_root}")
    print(f"Models: {args.models}")

    print("Loading ImageNet-100 ...")
    ds, eval_key = prepare_imagenet100(args.data_root)
    print(f"Splits: { {k: len(v) for k, v in ds.items()} }  eval={eval_key}")

    settings = TrainSettings(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        batch_cap=args.batch_cap,
    )
    fixed_batch = args.batch_size if args.batch_size > 0 else None

    for model_key in args.models:
        print("\n" + "=" * 96, flush=True)
        print(f"TRAINING: {model_key}", flush=True)
        print("=" * 96, flush=True)
        try:
            train_one(
                model_key,
                ds,
                eval_key,
                settings,
                args.runs_root,
                device,
                dtype,
                gpu_name,
                fixed_batch,
            )
        except Exception:
            traceback.print_exc()
            print_saved_results(args.runs_root, args.models)
            raise
    print_saved_results(args.runs_root, args.models)


if __name__ == "__main__":
    main()
