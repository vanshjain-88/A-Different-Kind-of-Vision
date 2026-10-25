#!/usr/bin/env python3
"""
Experiment map for ImageNet-100: every configuration in the code, what has run,
what is running and what is queued, with parameters and accuracies read from
runs/imagenet100/<name>/ (summary.json, history.csv, args.json) and logs/<name>.log.

    python3 analysis/plot_flowchart.py
writes runs/imagenet100/comparison/experiment_flowchart.png and .pdf.
Rerun at any time; finished runs fill in automatically.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]   # repository root
RUNS = ROOT / "runs" / "imagenet100"
LOGS = ROOT / "logs"
OUT = RUNS / "comparison"

# Parameter counts for runs that have not started yet (from building the model).
PLANNED_PARAMS = {
    "tok_learn_conv_db6_100": 6_771_300,
    "tok_learn_conv_db8_100": 6_773_092,
    "tok_learn_conv_db12_100": 6_778_212,
    "tok_pixels_100": 6_768_996,
    "hf_swin_tiny_100": 27_600_000,
    "hf_vit_base_100": 85_920_000,
}

COLUMNS = [
    ("Early wavelet models", "new.py", [
        ("Orthogonal Haar features", "haar_orth"),
        ("Overcomplete Haar features", "haar_over"),
    ]),
    ("Haar tokenizer + 12-block backbone", "haar_tok.py / haar_tok2.py", [
        "30-epoch ablations",
        ("Fixed Haar", "tok_fixed_conv"),
        ("Learned Haar", "tok_learn_conv"),
        ("Haar + modulus", "tok_learn_mod"),
        ("Haar, attention mixer", "tok_learn_attn"),
        "100 epochs",
        ("Haar, decimated", "tok_learn_conv_100"),
        ("Haar, undecimated", "tok_learn_conv_undecimated"),
    ]),
    ("Daubechies family, full model", "haar_tok_daub.py · MLP ratio 4", [
        ("Learned db2", "tok_learn_conv_db2_100"),
        ("Learned db4", "tok_learn_conv_db4_100"),
        ("Learned db6", "tok_learn_conv_db6_100"),
        ("Learned db8", "tok_learn_conv_db8_100"),
        ("Learned db12", "tok_learn_conv_db12_100"),
    ]),
    ("Wavelet vs pixels, MLP width", "haar_tok_daub.py · --tokenizer · --mlp-ratio", [
        "Learned db4 wavelet",
        ("MLP ratio 0", "tok_db4_learn_nomlp_100"),
        ("MLP ratio 1", "tok_db4_learn_mlp1_100"),
        ("MLP ratio 2", "tok_db4_learn_mlp2_100"),
        ("MLP ratio 4", "tok_learn_conv_db4_100"),
        "Raw pixel patches",
        ("MLP ratio 0", "tok_pixels_nomlp_100"),
        ("MLP ratio 1", "tok_pixels_mlp1_100"),
        ("MLP ratio 2", "tok_pixels_mlp2_100"),
        ("MLP ratio 4", "tok_pixels_100"),
    ]),
    ("Hugging Face baselines, from scratch", "hf_baselines.py", [
        ("ResNet-50", "hf_resnet50_100"),
        ("ConvNeXt-Tiny", "hf_convnext_tiny_100"),
        ("DeiT-Small", "hf_deit_small_100"),
        ("Swin-Tiny", "hf_swin_tiny_100"),
        ("ViT-Base", "hf_vit_base_100"),
    ]),
]

STYLE = {
    "done": {"face": "#E3F2E8", "edge": "#2E7D4F", "label": "Finished"},
    "running": {"face": "#FFF3CF", "edge": "#B8860B", "label": "Running"},
    "planned": {"face": "#E4ECF8", "edge": "#3D5A80", "label": "Queued / planned"},
    "idle": {"face": "#F0F0F0", "edge": "#9A9A9A", "label": "Available, not run"},
}
QUEUED = {
    "tok_learn_conv_db6_100": "queued",
    "tok_learn_conv_db8_100": "queued",
    "tok_learn_conv_db12_100": "queued",
    "tok_pixels_100": "planned",
}


def read_history(name):
    path = RUNS / name / "history.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def val_of(row):
    for key in ("val_top1", "val_acc"):
        if row.get(key) not in (None, ""):
            return float(row[key])
    return None


def params_from_log(name):
    path = LOGS / f"{name}.log"
    if not path.is_file():
        return None
    match = re.findall(r"params=([\d,]+)", path.read_text(errors="ignore"))
    return int(match[-1].replace(",", "")) if match else None


def run_info(name):
    summary_path = RUNS / name / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    args_path = RUNS / name / "args.json"
    args = json.loads(args_path.read_text()) if args_path.is_file() else {}
    status_path = RUNS / name / "status.json"
    legacy = json.loads(status_path.read_text()) if status_path.is_file() else {}
    history = read_history(name)
    vals = [v for v in (val_of(r) for r in history) if v is not None]
    target = int(args.get("epochs") or legacy.get("epochs") or 100)
    params = summary.get("params") or params_from_log(name) or PLANNED_PARAMS.get(name)

    if "test_top1" in summary:
        status = "done"
    elif history and len(history) >= target:
        status = "done"
    elif history:
        status = "running"
    elif name in QUEUED:
        status = "planned"
    else:
        status = "idle"

    if status == "done":
        val = summary.get("best_val_top1", max(vals) if vals else None)
        parts = [f"val {100 * val:.2f}%" if val is not None else None]
        parts.append(f"test {100 * summary['test_top1']:.2f}%" if "test_top1" in summary else "no test recorded")
        result = " · ".join(p for p in parts if p)
        if target != 100:
            result = f"{target} ep · " + result
    elif status == "running":
        result = f"running · epoch {len(history)}/{target}"
        if vals:
            result += f" · val {100 * max(vals):.2f}%"
    elif status == "planned":
        result = QUEUED[name]
    else:
        result = "not run"
    return status, params, result


def fmt_params(n):
    if not n:
        return "params unknown"
    if n >= 10_000_000:
        return f"{n / 1e6:.2f}M params"
    return f"{n:,} params"


def draw_box(ax, x, y, w, h, face, edge, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw,
    ))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    col_w, gap = 3.75, 0.32
    box_h, box_gap, sub_h = 0.92, 0.16, 0.42
    header_h = 0.95
    W = len(COLUMNS) * col_w + (len(COLUMNS) + 1) * gap

    def column_height(items):
        return sum(sub_h if isinstance(i, str) else box_h + box_gap for i in items)

    body = max(column_height(items) for _, _, items in COLUMNS)
    top_block = 2.45
    H = top_block + header_h + 0.45 + body + 0.9

    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(W / 2, H - 0.42, "ImageNet-100 experiments: configurations, parameters and accuracy",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#1F1F1F")

    data_w, data_h = 12.5, 1.15
    data_x, data_y = (W - data_w) / 2, H - 0.85 - data_h
    draw_box(ax, data_x, data_y, data_w, data_h, "#FFFFFF", "#444444", lw=1.4)
    ax.text(W / 2, data_y + data_h - 0.3, "Same data and protocol for every run",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#222222")
    ax.text(W / 2, data_y + 0.38,
            "ilee0022/ImageNet100 · train 117,000 · validation 13,000 · test 5,000 · 256×256 · batch 256 · AdamW lr 1e-3\n"
            "100 epochs unless noted · best-validation checkpoint evaluated once on the test set · accuracy is top-1",
            ha="center", va="center", fontsize=9.3, color="#333333", linespacing=1.5)

    header_y = data_y - 0.55 - header_h
    for ci, (title, script, items) in enumerate(COLUMNS):
        x = gap + ci * (col_w + gap)
        cx = x + col_w / 2
        ax.annotate("", xy=(cx, header_y + header_h), xytext=(W / 2, data_y),
                    arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0"))
        draw_box(ax, x, header_y, col_w, header_h, "#2F3B4C", "#2F3B4C")
        ax.text(cx, header_y + header_h - 0.33, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
        ax.text(cx, header_y + 0.27, script, ha="center", va="center", fontsize=8.6, color="#D8DEE8")

        spine_x = x + 0.16
        y = header_y - 0.3
        first_y = y
        last_mid = None
        for item in items:
            if isinstance(item, str):
                ax.text(x + 0.34, y - sub_h / 2 + 0.03, item, ha="left", va="center",
                        fontsize=9.2, fontweight="bold", color="#555555", style="italic")
                y -= sub_h
                continue
            label, name = item
            status, params, result = run_info(name)
            s = STYLE[status]
            bx, bw = x + 0.34, col_w - 0.34
            by = y - box_h
            draw_box(ax, bx, by, bw, box_h, s["face"], s["edge"])
            mid = by + box_h / 2
            ax.plot([spine_x, bx], [mid, mid], color="#9A9A9A", lw=1.0)
            last_mid = mid
            ax.text(bx + 0.13, by + box_h - 0.2, label, ha="left", va="center",
                    fontsize=9.6, fontweight="bold", color="#1F1F1F")
            ax.text(bx + bw - 0.12, by + box_h - 0.2, fmt_params(params), ha="right", va="center",
                    fontsize=8.6, color="#333333")
            ax.text(bx + 0.13, by + box_h / 2 - 0.03, name, ha="left", va="center",
                    fontsize=7.8, color="#666666", family="DejaVu Sans Mono")
            ax.text(bx + 0.13, by + 0.19, result, ha="left", va="center",
                    fontsize=9.2, color=s["edge"] if status != "done" else "#1E5A37",
                    fontweight="bold" if status == "done" else "normal")
            y -= box_h + box_gap
        if last_mid is not None:
            ax.plot([spine_x, spine_x], [header_y, last_mid], color="#9A9A9A", lw=1.0)

    lx = gap
    ly = 0.42
    for key in ("done", "running", "planned", "idle"):
        s = STYLE[key]
        draw_box(ax, lx, ly - 0.15, 0.42, 0.3, s["face"], s["edge"])
        ax.text(lx + 0.55, ly, s["label"], ha="left", va="center", fontsize=9.5, color="#333333")
        lx += 2.6

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"experiment_flowchart.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
