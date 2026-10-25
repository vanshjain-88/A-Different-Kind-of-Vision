#!/usr/bin/env bash
# Runs the MLP-width ablation one job after another through gpurun.
#
#   bash analysis/run_mlp_ablation.sh            # inside tmux, then detach
#
# Waits for tok_db4_learn_nomlp_100 if it is still running, skips any run that
# already has a test result, keeps going if a run fails, and at the end writes
#   runs/imagenet100/mlp_ablation/results.csv  and  results.md
# Progress goes to logs/mlp_ablation_queue.log; each run still writes its own
# logs/<name>.log and runs/imagenet100/<name>/ as usual.
# Rerunning the script is safe: finished runs are skipped and unfinished runs
# resume from their last.pt.

set -u
cd "$(dirname "$0")/.."          # repository root

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

RUNS_ROOT=runs/imagenet100
QUEUE_LOG=logs/mlp_ablation_queue.log
mkdir -p logs

WAIT_FOR=tok_db4_learn_nomlp_100

QUEUE=(
    "tok_db4_learn_mlp1_100|--wavelet db4 --mlp-ratio 1"
    "tok_pixels_nomlp_100|--tokenizer pixels --mlp-ratio 0"
    "tok_pixels_mlp1_100|--tokenizer pixels --mlp-ratio 1"
    "tok_db4_learn_mlp2_100|--wavelet db4 --mlp-ratio 2"
    "tok_pixels_mlp2_100|--tokenizer pixels --mlp-ratio 2"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

finished() {
    python3 - "$RUNS_ROOT/$1/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
sys.exit(0 if p.is_file() and "test_top1" in json.loads(p.read_text()) else 1)
PY
}

log "queue started: ${#QUEUE[@]} runs"

if pgrep -f "haar_tok_daub.py .*--name $WAIT_FOR" >/dev/null; then
    log "waiting for $WAIT_FOR to finish"
    while pgrep -f "haar_tok_daub.py .*--name $WAIT_FOR" >/dev/null; do
        sleep 60
    done
    log "$WAIT_FOR finished"
fi

for entry in "${QUEUE[@]}"; do
    name="${entry%%|*}"
    flags="${entry#*|}"
    if finished "$name"; then
        log "skip $name (already has a test result)"
        continue
    fi
    log "start $name: gpurun python3 ours/daubechies/haar_tok_daub.py $flags --name $name"
    start=$(date +%s)
    # shellcheck disable=SC2086
    gpurun python3 ours/daubechies/haar_tok_daub.py $flags --name "$name"
    status=$?
    mins=$(( ($(date +%s) - start) / 60 ))
    if [ $status -eq 0 ] && finished "$name"; then
        log "done  $name in ${mins} min"
    else
        log "FAILED $name (exit $status) after ${mins} min; see logs/$name.log, continuing"
    fi
done

python3 - "$RUNS_ROOT" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [
    ("Wavelet db4, learned", 0, "tok_db4_learn_nomlp_100"),
    ("Wavelet db4, learned", 1, "tok_db4_learn_mlp1_100"),
    ("Wavelet db4, learned", 2, "tok_db4_learn_mlp2_100"),
    ("Wavelet db4, learned", 4, "tok_learn_conv_db4_100"),
    ("Pixels", 0, "tok_pixels_nomlp_100"),
    ("Pixels", 1, "tok_pixels_mlp1_100"),
    ("Pixels", 2, "tok_pixels_mlp2_100"),
]
out_dir = root / "mlp_ablation"
out_dir.mkdir(parents=True, exist_ok=True)
table = []
for tokenizer, ratio, name in rows:
    s = root / name / "summary.json"
    d = json.loads(s.read_text()) if s.is_file() else {}
    table.append({
        "tokenizer": tokenizer,
        "mlp_ratio": ratio,
        "run": name,
        "params": d.get("params", ""),
        "best_val_top1": f"{100 * d['best_val_top1']:.2f}" if "best_val_top1" in d else "",
        "test_top1": f"{100 * d['test_top1']:.2f}" if "test_top1" in d else "",
    })
with (out_dir / "results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
lines = ["| Tokenizer | MLP ratio | Run | Params | Val top-1 | Test top-1 |", "|---|---:|---|---:|---:|---:|"]
for r in table:
    params = f"{r['params']:,}" if r["params"] != "" else "missing"
    val = f"{r['best_val_top1']}%" if r["best_val_top1"] else "missing"
    test = f"{r['test_top1']}%" if r["test_top1"] else "missing"
    lines.append(f"| {r['tokenizer']} | {r['mlp_ratio']} | {r['run']} | {params} | {val} | {test} |")
(out_dir / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "queue finished; results in $RUNS_ROOT/mlp_ablation/results.md"
