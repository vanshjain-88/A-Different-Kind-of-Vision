#!/usr/bin/env python3
"""
Hugging Face standard image classifiers on ImageNet-100, trained from scratch
under exactly the same protocol as haar_tok_daub.py.

Everything except the model is imported from haar_tok_daub.py: dataset, train /
validation / test split, augmentation, batch size, AdamW, LR schedule, label
smoothing, bf16 autocast, evaluation, best-validation checkpointing, the single
final test evaluation, and the logs/ + runs/ result files. Models are built from
their Hugging Face config only (random init, no pretrained weights), resized to
256x256 inputs and 100 classes.

Usage
    python3 baseline/hf_baselines.py --model resnet50
    python3 baseline/hf_baselines.py --model deit_small
    python3 baseline/hf_baselines.py --model convnext_tiny
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The training protocol (split, augmentation, schedule, evaluation, result files)
# is shared with our model so the comparison is like-for-like.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ours" / "daubechies"))
from haar_tok_daub import (
    IMAGE_SIZE, NUM_CLASSES, ROOT, Tee, append_shared_epoch_row, atomic_save,
    build_loaders, evaluate, load_splits, lr_at, read_history, set_seed,
    update_shared_final, write_history,
)

HF_MODELS = {
    "resnet50": "microsoft/resnet-50",
    "deit_small": "facebook/deit-small-patch16-224",
    "vit_base": "google/vit-base-patch16-224",
    "convnext_tiny": "facebook/convnext-tiny-224",
    "swin_tiny": "microsoft/swin-tiny-patch4-window7-224",
}
NO_DECAY_KEYWORDS = ("position_embeddings", "cls_token", "mask_token", "relative_position_bias_table")


class HFClassifier(nn.Module):
    """Wrap a Hugging Face model so forward(images) returns logits, like our models."""

    def __init__(self, repo_id: str, drop_path: float):
        super().__init__()
        from transformers import AutoConfig, AutoModelForImageClassification

        labels = {i: str(i) for i in range(NUM_CLASSES)}
        config = AutoConfig.from_pretrained(
            repo_id, id2label=labels, label2id={v: k for k, v in labels.items()},
        )
        if hasattr(config, "image_size"):
            config.image_size = IMAGE_SIZE
        self.drop_path = getattr(config, "drop_path_rate", None)
        if self.drop_path is not None:
            config.drop_path_rate = drop_path
            self.drop_path = drop_path
        self.net = AutoModelForImageClassification.from_config(config)

    def forward(self, x):
        return self.net(pixel_values=x).logits


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or any(k in n for k in NO_DECAY_KEYWORDS) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(HF_MODELS), required=True)
    p.add_argument("--name", default=None, help="run name (default: hf_<model>_100)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1, help="used only by models that support stochastic depth")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--workers", type=int, default=min(16, max(4, (os.cpu_count() or 8) - 2)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "imagenet100")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "imagenet100")
    args = p.parse_args()
    if args.name is None:
        args.name = f"hf_{args.model}_100"
    args.hf_repo = HF_MODELS[args.model]

    assert torch.cuda.is_available(), "CUDA not available (run with gpurun)"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16
    set_seed(args.seed)

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

    model = HFClassifier(args.hf_repo, args.drop_path).to(device)
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

    print(f"params={n_params:,} model={args.model} hf_repo={args.hf_repo} pretrained=False "
          f"image_size={IMAGE_SIZE} batch={args.batch_size} drop_path={model.drop_path}", flush=True)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    step = start_epoch * steps_per_epoch

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
            loss = F.cross_entropy(out.float(), y, label_smoothing=args.label_smoothing)
            loss.backward()
            opt.step()
            loss_acc += loss.detach().double() * y.numel()
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
        }
        history = [r for r in history if r["epoch"] != row["epoch"]] + [row]
        write_history(history, hist_p)
        append_shared_epoch_row(args.runs_root, args.name, row, args.batch_size)
        print(f"[{args.name}] {epoch + 1:03d}/{args.epochs:03d} train_loss={row['train_loss']:.4f} "
              f"train_top1={100 * row['train_top1']:.2f}% val_loss={val_loss:.4f} "
              f"val_top1={100 * val_top1:.2f}% best={100 * best:.2f}% | {row['img_per_sec']:.0f} img/s",
              flush=True)

        state = {"completed_epoch": epoch + 1, "best": best, "val_top1": val_top1,
                 "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}
        atomic_save(state, last_p)
        if val_top1 >= prev_best:
            atomic_save(state, best_p)
            print(f"[{args.name}] saved best checkpoint to {best_p}", flush=True)

    summary = {"name": args.name, "model": args.model, "hf_repo": args.hf_repo, "pretrained": False,
               "params": n_params, "best_val_top1": best}
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
