# A Different Kind of Vision

Can a fixed wavelet representation carry useful information for image classification?
This repository tests that on ImageNet-100 by replacing the usual learned patch embedding of a vision
backbone with a **wavelet tokenizer**, and comparing it against raw-pixel patches and against standard
Hugging Face models trained from scratch under exactly the same protocol.

## What the model is

```
image 3x256x256
  -> wavelet tokenizer   4-level decimated Haar / Daubechies transform, filters initialised to the
                         exact wavelet and fine-tuned (with an orthogonality penalty);
                         256 tokens (16x16 grid) of 768 coefficients each
  -> LayerNorm -> Linear(768 -> 256) + position embedding
  -> 12 blocks           depthwise 7x7 spatial mixing  +  per-token MLP (256 -> 1024 -> 256)
  -> mean over tokens -> Linear(256 -> 100)
```

About 6.77M parameters in total. The tokenizer itself is tiny (64 learnable parameters for Haar, 1,024 for db4,
9,216 for db12, and none if frozen). Roughly 93% of the parameters are the per-token MLPs in the 12 blocks,
which are identical in every wavelet variant, so the variants differ only in the tokenizer.

## Repository layout

```
baseline/
  hf_baselines.py            Hugging Face models from scratch: resnet50, deit_small, convnext_tiny, swin_tiny, vit_base
ours/
  haar/
    haar_tok.py              decimated Haar tokenizer + backbone
    haar_tok2.py             undecimated (a trous) Haar tokenizer + backbone
  daubechies/
    haar_tok_daub.py         Daubechies tokenizer db2/db4/db6/db8/db12, plus the ablation switches below
  early_wavelet/
    new.py                   first wavelet models (haar_orth, haar_over) and a ConvNeXt-V2 run
    vendor/wavevit_compat/   Wave-ViT helper code used by new.py
analysis/
  plot_comparison.py         comparison table and plots (ours vs ResNet-50 / DeiT-Small)
  plot_flowchart.py          experiment map with parameters and accuracies for every run
  run_mlp_ablation.sh        runs the MLP-width ablation queue unattended
logs/                        one <run>.log per run (stdout of the training script)
runs/imagenet100/            one folder per run: args.json, history.csv, summary.json
                             plus shared epoch_metrics.csv, final_results.csv/.json, comparison/, mlp_ablation/
docs/                        LaTeX report on the decimated Haar model
legacy_cifar10/              the earlier CIFAR-10 ViT project (self-contained, run from inside that folder)
data/                        created on first run (not in git)
```

Model checkpoints (`best.pt`, `last.pt`) are not in git because of their size; they are written to
`runs/imagenet100/<run>/` when you train.

## Setup

Python 3.12 and CUDA were used; library versions are in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # install torch/torchvision for your CUDA version first if needed
```

## Data

All scripts use the same dataset and split, the Hugging Face dataset
[`ilee0022/ImageNet100`](https://huggingface.co/datasets/ilee0022/ImageNet100) (100 classes):

| Split | Images | Use |
|---|---:|---|
| train | 117,000 | training |
| validation | 13,000 | model selection (best epoch) |
| test | 5,000 | evaluated once, on the best-validation checkpoint |

There is nothing to download by hand. The first run fetches the dataset (about 17 GB download) into
`data/imagenet100/` and every later run reuses it. To keep it elsewhere, pass `--data-root /path/to/dir`
(or symlink `data/`). Images are resized and cropped to 256x256; training uses random resized crop, horizontal
flip, RandAugment and random erasing, and no Mixup or CutMix.

## Training protocol (identical for every run)

100 epochs, batch 256, AdamW (lr 1e-3, weight decay 0.05), 5 warmup epochs then cosine decay, label smoothing 0.1,
drop-path 0.1 where the model supports it, seed 42, bf16 autocast with TF32 matmuls. After every epoch the validation
accuracy is logged and the best-validation checkpoint is kept. After the last epoch that checkpoint is evaluated once
on the test set. Runs resume automatically from `runs/imagenet100/<name>/last.pt` if you start the same command again.

## Running experiments

Run everything from the repository root. Each command writes `logs/<name>.log` and `runs/imagenet100/<name>/`.

```bash
# Our model: learned Daubechies tokenizer (db2, db4, db6, db8, db12)
python3 ours/daubechies/haar_tok_daub.py --wavelet db4 --name tok_learn_conv_db4_100

# Our model: Haar (decimated and undecimated)
python3 ours/haar/haar_tok.py  --name tok_learn_conv_100
python3 ours/haar/haar_tok2.py --name tok_learn_conv_undecimated

# Ablation switches (in haar_tok_daub.py)
--mlp-ratio 0|1|2|4         width of the per-token MLP; 0 removes it
--tokenizer pixels          raw 16x16 pixel patches instead of wavelet coefficients (the learned patch-embedding control)
--no-learn-filters          freeze the tokenizer at the exact wavelet
--mixer attn                self-attention instead of depthwise convolution
--epochs N                  shorter runs

# Baselines: Hugging Face models, random initialisation, same split and recipe
python3 baseline/hf_baselines.py --model resnet50
python3 baseline/hf_baselines.py --model deit_small
python3 baseline/hf_baselines.py --model convnext_tiny      # also: swin_tiny, vit_base

# MLP-width ablation queue (db4 and pixels at MLP ratio 1 and 2, then the table), unattended
bash analysis/run_mlp_ablation.sh

# Tables and plots from the saved runs
python3 analysis/plot_comparison.py          # runs/imagenet100/comparison/{comparison.md,val_top1_curves.png,test_val_bars.png}
python3 analysis/plot_flowchart.py           # runs/imagenet100/comparison/experiment_flowchart.png
```

## GPU configuration

- **Single GPU per run.** The scripts use `cuda:0` and do not use multi-GPU or DistributedDataParallel, so a job
  occupies one GPU. Pick the card with `CUDA_VISIBLE_DEVICES=1 python3 ...`.
- **Hardware used for every result here:** one NVIDIA A100 80 GB, on a node with two A100s. A CUDA GPU is required;
  the scripts stop with an assertion if none is found.
- **Precision:** bf16 autocast with TF32 enabled, so an Ampere-class GPU (A100, A6000, RTX 30/40 series, H100) or newer
  is expected. The wavelet tokenizer itself runs in fp32.
- **Memory:** if your GPU has less memory than an 80 GB A100, lower `--batch-size`. Keep in mind that the results above
  were obtained at batch 256, so a smaller batch is no longer the same protocol.
- **`gpurun`:** the runs were launched through `gpurun`, a GPU job broker specific to the institute cluster that
  enforces a weekly GPU-hour quota. It is not needed elsewhere. On the cluster, prefix any command with it,
  for example `gpurun python3 ours/daubechies/haar_tok_daub.py --wavelet db4`. `analysis/run_mlp_ablation.sh` calls
  `gpurun` internally; replace it with plain `python3` in that file if you do not have it.
- **Time on an A100:** about 1.5 h for Haar, 1.6 h for db2, 1.8 h for db4, 2.1 h for db6, 2.6 h for db8, 4.2 h for
  db12 (longer filters make the tokenizer slower; accuracy does not improve). ResNet-50 takes about 3.6 h,
  DeiT-Small about 2.2 h.

## Results (ImageNet-100 test top-1, best-validation checkpoint, one seed)

| Model | Params | Val | Test |
|---|---:|---:|---:|
| ResNet-50 (Hugging Face, scratch) | 23.71M | 85.53% | **82.34%** |
| ConvNeXt-Tiny (Hugging Face, scratch) | 27.90M | 81.85% | 78.44% |
| DeiT-Small (Hugging Face, scratch) | 21.73M | 71.73% | 66.80% |
| Ours, Haar decimated | 6.77M | 78.09% | 73.80% |
| Ours, Haar undecimated | 6.77M | 76.41% | 72.72% |
| Ours, db2 | 6.77M | 78.78% | 74.16% |
| Ours, db4 | 6.77M | 77.95% | **74.44%** |
| Ours, db6 | 6.77M | 78.22% | 73.82% |
| Ours, db8 | 6.77M | 77.41% | 73.86% |
| Ours, db12 | 6.78M | 77.73% | 73.18% |

**Wavelet versus raw pixels, by MLP width** (learned tokenizer in both; test top-1):

| MLP ratio | Params | db4 wavelet | Raw pixels |
|---:|---:|---:|---:|
| 0 | 0.45M | 36.52% | 33.86% |
| 1 | 2.04M | 73.06% | 73.16% |
| 2 | 3.62M | 73.84% | 73.48% |
| 4 | 6.77M | 74.44% | not run |

The wavelet tokenizer helps clearly only when the network has no MLP; with an MLP the two inputs are within the
sampling noise of the 5,000-image test set (about +-0.6 points). The Haar-to-db8 variants are also within that noise
of each other.

## Notes and caveats

- All numbers are single-seed. Differences below about 1 point between our variants should not be read as a ranking.
- The Hugging Face baselines have 3 to 4 times more parameters and about 3 times more compute per image than our
  models (forward pass: 3.4 GFLOPs for ours, 10.7 for ResNet-50, 11.1 for DeiT-Small). ResNet-50 and DeiT-Small do not
  support stochastic depth, so the 0.1 drop-path setting only applies to ConvNeXt and Swin.
- The 30-epoch runs (`tok_fixed_conv`, `tok_learn_conv`, `tok_learn_mod`, `tok_learn_attn`) are early ablations and are
  not comparable with the 100-epoch runs. The two `new.py` runs (`haar_orth`, `haar_over`, 30 epochs, 75,100 parameters)
  only record validation accuracy.
- `runs/imagenet100/` also contains three unfinished leftovers: `smoke` (a 1-epoch test), `haar_tok2` (stopped after
  2 epochs) and `convnextv2` (stopped after 1 epoch).
- Absolute paths inside `args.json` and `final_results.*` point to the machine the runs were produced on.
- `legacy_cifar10/` is the earlier CIFAR-10 work and has its own README and requirements.
