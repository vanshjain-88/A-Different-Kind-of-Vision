# Project Structure and Code Reference

The baseline ViT described here has a `vit/wavelets/` package containing the
fixed `bior2.2` and `coif1` transforms, plus a dual-branch input path in
`PatchEmbedding`. See `README.md` for the wavelet architecture and tensor
contracts.

## 1. Directory Structure

The source tree follows the same stages as the ViT architecture:

```text
BTP/
├── vit/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── wavelets/
│   │   ├── __init__.py
│   │   └── fixed_dwt.py
│   ├── patch_embedding/
│   │   ├── __init__.py
│   │   └── patch_embedding.py
│   ├── transformer_encoder/
│   │   ├── __init__.py
│   │   └── transformer_encoder.py
│   └── classification_head/
│       ├── __init__.py
│       └── classification_head.py
├── training/
│   ├── __init__.py
│   └── train.py
├── tests/
│   ├── test_patch_embedding.py
│   ├── test_transformer_encoder.py
│   └── test_training.py
├── inspect_cifar_patchification.py
└── docs/
    ├── architecture_flow.md
    └── code_reference.md
```

Each architecture directory owns exactly one conceptual stage:

```text
patch_embedding → transformer_encoder → classification_head
```

The short `__init__.py` files expose useful classes so callers do not need to
know the internal filename.

## 2. Public Imports

Most code only needs:

```python
from vit import ViTConfig, VisionTransformer
```

Individual components remain available for learning and testing:

```python
from vit.patch_embedding import PatchEmbedding
from vit.transformer_encoder import MultiHeadSelfAttention
from vit.transformer_encoder import MLPBlock, EncoderBlock, TransformerEncoder
from vit.classification_head import ClassificationHead
```

## 3. `vit/config.py`

### `ViTConfig`

The frozen data class contains every model setting:

- `image_size`
- `patch_size`
- `in_channels`
- `embedding_dim`
- `num_layers`
- `num_heads`
- `mlp_dim`
- `dropout`
- `attention_dropout`
- `layer_norm_eps`
- `num_classes`

`__post_init__` checks positive dimensions, patch divisibility, attention-head
divisibility, valid dropout probabilities, and positive LayerNorm epsilon.

Calculated properties:

```text
patches_per_side = image_size // patch_size
num_patches      = patches_per_side²
patch_dim        = in_channels × patch_size²
```

## 4. `vit/patch_embedding/`

### `PatchEmbedding`

`patch_embedding.py` implements everything before the Encoder:

```text
Image
→ patchify
→ linear projection
→ prepend CLS
→ add positions
→ Encoder input tokens
```

### Methods

- `reset_parameters`: initializes projection, CLS, and position parameters.
- `patchify`: rearranges `[B,C,H,W]` into `[B,N,C×P×P]`.
- `project_patches`: applies `Linear(patch_dim, embedding_dim)`.
- `prepend_cls`: expands and prepends the shared learnable CLS token.
- `add_position`: adds learned `[1,N+1,D]` positions.
- `forward`: calls those four stages in order.

## 5. `vit/transformer_encoder/`

`transformer_encoder.py` keeps the paper's Encoder operations visible instead
of hiding them behind `nn.MultiheadAttention`.

### `MultiHeadSelfAttention`

Projects query, key, and value; splits them into heads; computes scaled
dot-product attention; joins the heads; and applies the output projection.

### `MLPBlock`

Implements:

```text
Linear(D, mlp_dim) → GELU → Dropout
→ Linear(mlp_dim, D) → Dropout
```

### `EncoderBlock`

Implements the paper's two pre-LayerNorm residual equations:

```python
tokens = tokens + self.attention(self.norm1(tokens))
tokens = tokens + self.mlp(self.norm2(tokens))
```

### `TransformerEncoder`

Applies input dropout, repeats `EncoderBlock` `num_layers` times, and applies
the final LayerNorm.

## 6. `vit/classification_head/`

### `ClassificationHead`

Selects the encoded CLS token and applies the paper's MLP head:

```text
[B,N+1,D]
→ select [:,0]
→ Linear(D,D)
→ tanh
→ Linear(D,num_classes)
→ [B,num_classes]
```

The output contains raw logits, not probabilities.

## 7. `vit/model.py`

### `ViTInputPipeline`

Runs only `PatchEmbedding`. It remains useful for inspecting the Encoder input
boundary.

### `VisionTransformer`

Connects the three architecture directories:

```python
self.patch_embedding = PatchEmbedding(config)
self.encoder = TransformerEncoder(config)
self.head = ClassificationHead(config)
```

`forward_features(images)` returns encoded `[B,N+1,D]` tokens.

`forward(images)` passes the encoded tokens through the head and returns
`[B,num_classes]` logits.

## 8. `training/train.py`

This module is deliberately separate from the model. Run it from the project
root:

```bash
python -m training.train
```

### `parse_args`

Defines command-line arguments for training and every model dimension. This
allows configuration changes without editing Python files:

```bash
python -m training.train \
  --patch-size 2 \
  --embedding-dim 256 \
  --num-layers 8 \
  --num-heads 8 \
  --mlp-dim 1024 \
  --batch-size 64
```

Use this to list every option:

```bash
python -m training.train --help
```

### `choose_device`

With `--device auto`, selects CUDA first, then Apple MPS, then CPU. Kaggle
automatically uses its CUDA GPU.

### `create_dataloaders`

Splits CIFAR-10's original 50,000 training examples into 45,000 training and
5,000 validation examples using a seeded random split. It also creates a
separate loader for the original 10,000 test examples. Images are converted
to tensors and normalized with CIFAR-10 channel statistics. No augmentation
is added.

The validation loader is used for model selection. The test loader is not
used during training or checkpoint selection.

To use the dataset already extracted in this repository:

```bash
python -m training.train --data-dir . --no-download
```

On Kaggle, the default downloads CIFAR-10 into `data/`.

### `train_one_epoch`

For each training batch:

1. moves images and labels to the device;
2. clears old gradients;
3. computes logits;
4. computes cross-entropy loss;
5. runs backpropagation;
6. updates parameters with Adam;
7. accumulates loss and accuracy.

### `evaluate`

Uses `model.eval()` and `torch.no_grad()` to calculate test loss and accuracy
without updating parameters.

### `main`

Creates the configuration, model, DataLoaders, cross-entropy loss, and Adam
optimizer. It trains for the requested epochs and saves the checkpoint with
the best validation accuracy. After training, it loads that checkpoint and
evaluates the untouched test set once.

The checkpoint contains:

- epoch number;
- complete model configuration;
- model parameters;
- optimizer state;
- validation accuracy.

Default checkpoint:

```text
training/checkpoints/best.pt
```

## 9. Training Commands

Default CIFAR-10 training:

```bash
python -m training.train
```

Short local test:

```bash
python -m training.train \
  --data-dir . \
  --no-download \
  --epochs 1 \
  --batch-size 128
```

Typical Kaggle command:

```bash
python -m training.train \
  --device cuda \
  --epochs 100 \
  --batch-size 128
```

## 10. Inspection and Tests

`inspect_cifar_patchification.py` loads four real images, verifies exact
patch reconstruction, and prints shapes through the classifier:

```bash
python inspect_cifar_patchification.py
```

Run all unit tests:

```bash
python -m unittest discover -s tests -v
```

The tests cover patch values and shapes, configuration validation,
self-attention, MLPs, residual paths, final normalization, classification,
complete-model output, and a synthetic training/evaluation step.
