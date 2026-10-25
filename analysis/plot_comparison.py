#!/usr/bin/env python3
"""
Compare the Hugging Face industry baselines with our full-length ImageNet-100 runs.

Reads each run's summary.json and history.csv under runs/imagenet100 and writes:
  comparison/comparison.csv
  comparison/comparison.md
  comparison/val_top1_curves.png   (and .pdf)
  comparison/test_val_bars.png     (and .pdf)

By default the industry pair is ResNet-50 and DeiT-Small, and "ours" is the four
full-model 100-epoch tokenizers (db4, db2, Haar decimated, Haar undecimated).
Pass --ours <run names> for any other set, e.g. the db6/db8/db12 runs.

    python3 analysis/plot_comparison.py
    python3 analysis/plot_comparison.py --ours tok_learn_conv_db4_100 tok_learn_conv_db8_100 tok_learn_conv_db12_100
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]   # repository root
RUNS = ROOT / "runs" / "imagenet100"

INDUSTRY_DEFAULT = ("hf_resnet50_100", "hf_deit_small_100")
OURS_DEFAULT = ("tok_learn_conv_db4_100", "tok_learn_conv_db2_100",
                "tok_learn_conv_100", "tok_learn_conv_undecimated")
MIN_EPOCHS = 90

LABELS = {
    "tok_learn_conv_db4_100": "Daubechies-4 (ours)",
    "tok_learn_conv_db2_100": "Daubechies-2 (ours)",
    "tok_learn_conv_100": "Haar decimated (ours)",
    "tok_learn_conv_undecimated": "Haar undecimated (ours)",
    "tok_learn_conv_db6_100": "Daubechies-6 (ours)",
    "tok_learn_conv_db8_100": "Daubechies-8 (ours)",
    "tok_learn_conv_db12_100": "Daubechies-12 (ours)",
    "hf_resnet50_100": "ResNet-50",
    "hf_deit_small_100": "DeiT-Small",
}
# Ours share a blue-teal ramp; the two industry models are warm and dashed.
OURS_COLORS = ["#0B6E4F", "#1B9AAA", "#3D5A80", "#7EB6D9"]
INDUSTRY_COLORS = {"hf_resnet50_100": "#E07A3D", "hf_deit_small_100": "#C44536"}


def label_of(name: str) -> str:
    return LABELS.get(name, name)


def load_history(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("epoch", "train_top1", "val_top1", "val_loss", "train_loss"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def load_run(runs_root: Path, name: str) -> dict:
    run_dir = runs_root / name
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
    if not summary_path.is_file() or not history_path.is_file():
        raise SystemExit(f"missing summary.json or history.csv for {name} in {run_dir}")
    summary = json.loads(summary_path.read_text())
    history = load_history(history_path)
    if not history or "val_top1" not in history[0]:
        raise SystemExit(f"{name} history has no val_top1 column")
    best = max(history, key=lambda r: r["val_top1"])
    return {
        "name": name,
        "label": label_of(name),
        "group": "industry" if name.startswith("hf_") else "ours",
        "params": int(summary.get("params", 0)),
        "best_val_top1": float(summary.get("best_val_top1", best["val_top1"])),
        "test_top1": float(summary["test_top1"]) if "test_top1" in summary else None,
        "best_epoch": int(best["epoch"]),
        "epochs": int(history[-1]["epoch"]),
        "history": history,
    }


def discover_ours(runs_root: Path, top: int) -> list[str]:
    found = []
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        name = summary_path.parent.name
        if name.startswith("hf_") or name in INDUSTRY_DEFAULT:
            continue
        summary = json.loads(summary_path.read_text())
        if "test_top1" not in summary:
            continue
        history_path = summary_path.parent / "history.csv"
        if not history_path.is_file():
            continue
        with history_path.open(newline="") as f:
            n_epochs = sum(1 for _ in f) - 1
        if n_epochs < MIN_EPOCHS:
            continue
        found.append((float(summary["test_top1"]), name))
    found.sort(reverse=True)
    if not found:
        raise SystemExit(f"no full-length runs (>= {MIN_EPOCHS} epochs) found under {runs_root}")
    return [name for _, name in found[:top]]


def write_table(runs: list[dict], out_dir: Path) -> None:
    header = ["model", "group", "params", "params_m", "best_epoch", "epochs",
              "best_val_top1", "test_top1"]
    ordered = sorted(runs, key=lambda r: (-(r["group"] == "ours"), -(r["test_top1"] or -1)))
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for run in ordered:
            writer.writerow({
                "model": run["label"],
                "group": run["group"],
                "params": run["params"],
                "params_m": f"{run['params'] / 1e6:.2f}",
                "best_epoch": run["best_epoch"],
                "epochs": run["epochs"],
                "best_val_top1": f"{100 * run['best_val_top1']:.2f}",
                "test_top1": "" if run["test_top1"] is None else f"{100 * run['test_top1']:.2f}",
            })

    md_path = out_dir / "comparison.md"
    lines = [
        "| Model | Group | Params | Best epoch | Val top-1 | Test top-1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in ordered:
        test = "—" if run["test_top1"] is None else f"{100 * run['test_top1']:.2f}%"
        lines.append(
            f"| {run['label']} | {run['group']} | {run['params'] / 1e6:.2f}M | "
            f"{run['best_epoch']} | {100 * run['best_val_top1']:.2f}% | {test} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def colors_for(runs: list[dict]) -> dict[str, str]:
    ours = [r["name"] for r in runs if r["group"] == "ours"]
    colors = {}
    for i, name in enumerate(ours):
        colors[name] = OURS_COLORS[i % len(OURS_COLORS)]
    for run in runs:
        if run["group"] == "industry":
            colors[run["name"]] = INDUSTRY_COLORS.get(run["name"], "#888888")
    return colors


def plot_curves(runs: list[dict], colors: dict[str, str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for run in runs:
        h = run["history"]
        ax.plot(
            [row["epoch"] for row in h],
            [100 * row["val_top1"] for row in h],
            color=colors[run["name"]],
            linewidth=2.0,
            linestyle="--" if run["group"] == "industry" else "-",
            label=f"{run['label']}  ({run['params'] / 1e6:.1f}M)",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 (%)")
    ax.set_title("ImageNet-100 validation accuracy")
    ax.set_xlim(1, max(r["epochs"] for r in runs))
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"val_top1_curves.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_bars(runs: list[dict], colors: dict[str, str], out_dir: Path) -> None:
    ordered = sorted(runs, key=lambda r: r["test_top1"] or -1)
    labels = [f"{r['label']}\n{r['params'] / 1e6:.1f}M" for r in ordered]
    val = [100 * r["best_val_top1"] for r in ordered]
    test = [0.0 if r["test_top1"] is None else 100 * r["test_top1"] for r in ordered]
    bar_colors = [colors[r["name"]] for r in ordered]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = list(range(len(ordered)))
    height = 0.36
    ax.barh([i + height / 2 for i in y], val, height=height, color="#D9D9D9")
    ax.barh([i - height / 2 for i in y], test, height=height, color=bar_colors)
    for i, (v, t) in enumerate(zip(val, test)):
        ax.text(t + 0.4, i - height / 2, f"{t:.2f}", va="center", ha="left", fontsize=8, color="#222222")
        ax.text(v + 0.4, i + height / 2, f"{v:.2f}", va="center", ha="left", fontsize=8, color="#666666")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("Best validation checkpoint, evaluated once on the test set")
    ax.set_xlim(0, max(val + test) + 10)
    style_axes(ax)
    ax.legend(
        handles=[
            Patch(facecolor="#D9D9D9", label="Best validation"),
            Patch(facecolor="#4A4A4A", label="Test"),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"test_val_bars.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=RUNS)
    p.add_argument("--out", type=Path, default=None, help="output directory (default: <runs-root>/comparison)")
    p.add_argument("--top", type=int, default=None,
                   help="instead of the default four, keep the top N full-length runs by test top-1 (picks up ablations too)")
    p.add_argument("--ours", nargs="*", default=None, help="run folder names (default: Daubechies-4, Daubechies-2, Haar decimated, Haar undecimated)")
    p.add_argument("--industry", nargs="*", default=list(INDUSTRY_DEFAULT))
    args = p.parse_args()

    out_dir = args.out or (args.runs_root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ours:
        ours_names = args.ours
    elif args.top:
        ours_names = discover_ours(args.runs_root, args.top)
    else:
        ours_names = list(OURS_DEFAULT)
    missing = [n for n in list(args.industry) + ours_names if not (args.runs_root / n).is_dir()]
    if missing:
        raise SystemExit(f"run folders not found: {', '.join(missing)}")

    runs = [load_run(args.runs_root, name) for name in list(args.industry) + ours_names]
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
    })
    write_table(runs, out_dir)
    colors = colors_for(runs)
    plot_curves(runs, colors, out_dir)
    plot_bars(runs, colors, out_dir)


if __name__ == "__main__":
    main()
