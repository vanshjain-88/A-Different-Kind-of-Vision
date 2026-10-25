# Wavelet Vision Transformer

This project implements a small Vision Transformer for CIFAR-10 with three
selectable tokenizers for controlled comparisons:

- `patch`: ordinary flattened RGB patches.
- `biorthogonal`: an RGB branch fused with a one-level `bior2.2` wavelet branch.
- `coiflet`: an RGB branch fused with a one-level `coif1` wavelet branch.

## Architecture

```text
RGB image [B, 3, 32, 32]
       |                         \
       |                  bior2.2 or coif1 DWT
       |                         / [B, 12, 16, 16]
       v                        v
RGB patchify                Wavelet patchify
       |                        |
       v                        v
RGB projection              Wavelet projection
       |                        |
       +------ concatenate ----+
                    |
              Fusion Linear
                    |
             CLS + positions
                    |
          Transformer Encoder
                    |
          Tanh classification head
                    |
                Class logits
```

For the default `32x32` images and `patch_size=4`, the RGB branch uses
`4x4` patches and the wavelet branch uses `2x2` patches on the `16x16`
subbands. Both branches therefore produce 64 spatial tokens. Each branch is
projected independently before concatenation and fusion, preserving separate
learned representations for RGB and wavelet information.

Both transforms produce four subbands per input channel in this order:
`LL`, `LH`, `HL`, `HH`. Circular boundary padding keeps the one-level output
at exactly half the input height and width. The filters are fixed while the
RGB projection, wavelet projection, and fusion layer are learned.

## Project layout

- `vit/wavelets/`: fixed one-level biorthogonal and Coiflet DWT versions.
- `vit/patch_embedding/`: RGB and wavelet patching, projection, fusion, CLS,
  and positional embeddings.
- `vit/transformer_encoder/`: pre-LayerNorm ViT encoder blocks.
- `vit/classification_head/`: CLS-token MLP classifier.
- `training/train.py`: CIFAR-10 data loading and training loop.
- `tests/`: shape, numerical, encoder, and training tests.

## Running

Create and activate the project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The environment can be left with `deactivate`. Alternatively, every command
can be run without activation by replacing `python` with `.venv/bin/python`.

Run the test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Train from the project root:

```bash
python -m training.train --data-dir data
```

The model accepts ordinary normalized CIFAR tensors with shape
`[B, 3, 32, 32]` and returns logits with shape `[B, 10]`.

With the default `--device auto`, training uses CUDA when available, then
Apple MPS, then CPU. CIFAR-10 is downloaded into `data/` unless
`--no-download` is supplied. Use `--help` to inspect all training and model
options.

## Outputs

The trainer saves the best validation checkpoint to
`training/checkpoints/best.pt` by default. It contains the epoch, model
configuration, model parameters, optimizer state, and validation accuracy.
The path can be changed with `--checkpoint`.

The trainer also writes two CSV files by default:

- `training/checkpoints/epoch_metrics.csv`: one row per epoch with `epoch`,
       `train_loss`, `train_accuracy`, `validation_loss`, and
       `validation_accuracy`.
- `training/checkpoints/final_test_metrics.csv`: one row for the final test
       evaluation with `checkpoint_epoch`, `test_loss`, and `test_accuracy`.

Override their locations with `--metrics-csv` and `--test-csv`. Both files are
recreated for each training run.

## Controlled tokenizer comparison

Keep every non-tokenizer argument and seed fixed. Use separate output paths so
runs do not overwrite one another:

```bash
python -m training.train --tokenizer patch \
  --checkpoint training/checkpoints/patch.pt \
  --metrics-csv training/checkpoints/patch.csv \
  --test-csv training/checkpoints/patch-test.csv

python -m training.train --tokenizer biorthogonal \
  --checkpoint training/checkpoints/biorthogonal.pt \
  --metrics-csv training/checkpoints/biorthogonal.csv \
  --test-csv training/checkpoints/biorthogonal-test.csv

python -m training.train --tokenizer coiflet \
  --checkpoint training/checkpoints/coiflet.pt \
  --metrics-csv training/checkpoints/coiflet.csv \
  --test-csv training/checkpoints/coiflet-test.csv
```

## Recommended-wavelet experiment

`experiments/recommended_wavelet/` isolates a more substantial tokenizer
variant without changing the main comparison. It:

- applies Shifted Patch Tokenization (SPT) to the pixel stream;
- converts RGB to YCbCr, downsamples Cb/Cr, then applies a selectable Haar or
  `bior2.2` DWT;
- projects and normalizes LL and detail bands separately, then combines them
  with learned weights;
- keeps pixel and wavelet embeddings as separate sequence tokens with learned
  type embeddings instead of linearly fusing them.

This gives the Encoder `129` tokens for CIFAR-10: one CLS token, 64 pixel
tokens, and 64 wavelet tokens. Run the two controlled versions with:

```bash
python -m experiments.recommended_wavelet.train --wavelet haar
python -m experiments.recommended_wavelet.train --wavelet biorthogonal
```

Results are kept out of `training/checkpoints/` and saved under
`experiments/recommended_wavelet/results/haar/` or
`experiments/recommended_wavelet/results/biorthogonal/`.