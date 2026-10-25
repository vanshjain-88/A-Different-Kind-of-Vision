# Vision Transformer Architecture Flow

The current implementation extends the baseline input path with a selectable
one-level `bior2.2` or `coif1` DWT. The RGB and wavelet branches are patched
and projected separately, fused into one token sequence, and then passed
through the same Transformer and classifier described below. See the
repository `README.md` for the complete updated flow and tensor shapes.

## 1. Implemented Scope

The project now implements the original Vision Transformer from the input
image through its MLP classification head:

```text
Image
  ↓
Non-overlapping flattened patches
  ↓
Linear patch projection
  ↓
Prepend learnable CLS token
  ↓
Add learnable 1D positional embeddings
  ↓
Transformer Encoder blocks
  ↓
Final LayerNorm
  ↓
Encoded token sequence
  ↓
Select encoded CLS token
  ↓
MLP Head: Linear → tanh → Linear
  ↓
Class logits
```

It does not include a CNN stem, relative positions, rotary positions,
stochastic depth, or other later ViT improvements.

The implementation follows Section 3.1 and Equations 1-4 of
`2010.11929v2.pdf`. Self-attention follows Equations 5-8 in Appendix A.

## 2. Complete CIFAR-10 Shape Flow

Using the default educational configuration:

```text
image_size     = 32
patch_size     = 4
in_channels    = 3
embedding_dim  = 128
num_heads      = 8
mlp_dim        = 512
num_layers     = 6
```

the complete shape flow is:

```text
Images
[B, 3, 32, 32]
        │
        │ split into an 8 × 8 patch grid
        ▼
Flattened patches
[B, 64, 48]
        │
        │ Linear(48, 128)
        ▼
Patch embeddings
[B, 64, 128]
        │
        │ prepend [B, 1, 128] CLS tokens
        ▼
Tokens with CLS
[B, 65, 128]
        │
        │ add [1, 65, 128] learned positions
        ▼
Encoder input
[B, 65, 128]
        │
        │ six Encoder blocks
        ▼
Final LayerNorm
[B, 65, 128]
        │
        ▼
Encoded tokens
[B, 65, 128]
        │
        │ select sequence position zero
        ▼
Encoded CLS representation
[B, 128]
        │
        │ Linear(128, 128) → tanh → Linear(128, 10)
        ▼
Class logits
[B, 10]
```

The Encoder preserves the sequence shape. It changes the token values by
allowing tokens to exchange information.

## 3. Patch and Token Preparation

For image height `H`, width `W`, channels `C`, and patch size `P`:

```text
N = HW / P²
patch_dim = P²C
```

Each real image patch is flattened and projected into the Encoder's constant
hidden dimension `D`:

```text
xₚⁱ ∈ R^(P²C)
xₚⁱE ∈ R^D
```

A shared learnable CLS embedding is placed at sequence index zero. A
learnable 1D position vector is added to the CLS token and every patch token.
This is Equation 1:

```text
z₀ = [x_class; xₚ¹E; xₚ²E; ...; xₚᴺE] + E_pos
```

For CIFAR-10, `z₀` has shape `[B, 65, 128]`.

## 4. One Transformer Encoder Block

Every block contains exactly two sublayers:

1. Multi-head self-attention.
2. A two-layer MLP with GELU.

Both use pre-LayerNorm and a residual connection:

```text
Input z
  │
  ├─────────────────────────────┐
  │                             │ residual
  ▼                             │
LayerNorm                       │
  ▼                             │
Multi-Head Self-Attention       │
  ▼                             │
Add ◄───────────────────────────┘
  │
  │ z'
  ├─────────────────────────────┐
  │                             │ residual
  ▼                             │
LayerNorm                       │
  ▼                             │
MLP: Linear → GELU → Linear     │
  ▼                             │
Add ◄───────────────────────────┘
  │
  ▼
Block output
```

This is precisely the ordering in the paper:

```text
z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁
zₗ  = MLP(LN(z'ₗ)) + z'ₗ
```

LayerNorm is therefore applied before attention and before the MLP. The
residual path bypasses each sublayer and is added afterward.

## 5. Multi-Head Self-Attention

Let:

```text
B  = batch size
S  = N + 1 sequence positions
D  = embedding dimension
h  = number of attention heads
Dh = D / h dimensions per head
```

### 5.1 Query, key, and value projection

One linear layer computes query, key, and value together:

```text
[B, S, D]
  ↓ Linear(D, 3D)
[B, S, 3D]
  ↓ split Q, K, V and split heads
Q, K, V: [B, h, S, Dh]
```

This combined layer represents the paper's trainable `U_qkv` matrix. Combining
the calculations is mathematically the same as using separate query, key, and
value layers.

### 5.2 Attention scores

Within every head, each query token is compared with every key token:

```text
Q @ Kᵀ

[B, h, S, Dh] @ [B, h, Dh, S]
  ↓
[B, h, S, S]
```

The scores are divided by `sqrt(Dh)`:

```text
scores = QKᵀ / sqrt(Dh)
```

Scaling prevents dot products from becoming too large as the head dimension
increases.

### 5.3 Softmax

Softmax is applied across the final `S` dimension:

```text
A = softmax(scores)
```

For one query token, its `S` attention weights sum to one. There is no
attention mask, so every image token can attend to every other token,
including CLS.

### 5.4 Weighted values and joined heads

The attention weights mix the value vectors:

```text
A @ V

[B, h, S, S] @ [B, h, S, Dh]
  ↓
[B, h, S, Dh]
```

The heads are transposed and concatenated:

```text
[B, h, S, Dh] → [B, S, h × Dh] → [B, S, D]
```

A final learned output projection mixes information across the concatenated
heads. The output remains `[B, S, D]`, allowing the residual addition.

## 6. MLP Block

Each token independently passes through the same two-layer MLP:

```text
[B, S, D]
  ↓ Linear(D, mlp_dim)
[B, S, mlp_dim]
  ↓ GELU
[B, S, mlp_dim]
  ↓ Linear(mlp_dim, D)
[B, S, D]
```

The MLP changes features inside each token. Attention mixes information
between tokens; the MLP then transforms each token's resulting features.

The paper's ViT variants use `mlp_dim = 4D`. The default educational
configuration follows that ratio: `512 = 4 × 128`.

## 7. Residual Connections

A residual connection adds a sublayer's input to its output:

```text
output = input + transformation(input)
```

This preserves a direct path through deep networks and allows a block to
learn a change to the current representation instead of replacing it
entirely.

Both attention and the MLP have their own residual connection, exactly as in
Equations 2 and 3.

## 8. Encoder Stack and Final LayerNorm

The same block structure is repeated `num_layers` times:

```text
z₀ → EncoderBlock₁ → EncoderBlock₂ → ... → EncoderBlockL → LayerNorm
```

Every layer has separate learnable parameters. The blocks do not share
weights.

After the final block, a final LayerNorm is applied. Equation 4 defines the
image representation as:

```text
y = LN(zᴸ₀)
```

where sequence index zero is the CLS token. The implementation normalizes the
whole sequence, as the original ViT implementation does, so:

```text
encoded_tokens[:, 0]
```

is exactly the normalized CLS representation consumed by the classification
head.

## 9. MLP Classification Head

The paper uses the encoded CLS token as the image representation:

```text
encoded_tokens[:, 0] → [B, D]
```

For pre-training, the paper describes an MLP head with one hidden layer and a
`tanh` activation:

```text
[B, D]
  ↓ Linear(D, D)
  ↓ tanh
  ↓ Linear(D, num_classes)
[B, num_classes]
```

The final values are logits. They are raw class scores, so softmax is not
applied inside the model. A training loss such as `CrossEntropyLoss` applies
the required normalization internally.

The final classifier weights and bias are initialized to zero, matching the
original ViT implementation. The hidden pre-logits layer uses Xavier
initialization.

## 10. Dropout Placement

The paper's experiment details apply dropout:

- after adding patch and positional embeddings;
- after the attention output projection;
- inside the MLP after GELU and after its second linear layer;
- optionally to attention weights through `attention_dropout`.

Dropout is active only in training mode:

```python
model.train()
```

It is disabled by:

```python
model.eval()
```

No dropout is added to residual paths, LayerNorm, or the QKV projection.

## 11. Configuration and Paper Model Sizes

The Encoder equations are independent of model size. The paper reports:

```text
ViT-Base:  layers=12, D=768,  mlp_dim=3072, heads=12
ViT-Large: layers=24, D=1024, mlp_dim=4096, heads=16
ViT-Huge:  layers=32, D=1280, mlp_dim=5120, heads=16
```

This project keeps the earlier CIFAR-10-friendly dimensions so it can run on
small hardware. That changes capacity, not the architecture or equations.

For a paper-sized ViT-Base/16 configuration, use:

```python
config = ViTConfig(
    image_size=224,
    patch_size=16,
    in_channels=3,
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_dim=3072,
    dropout=0.1,
    attention_dropout=0.0,
    num_classes=1000,
)
```

## 12. Model Outputs

Calling the model normally returns class logits:

```text
[B, num_classes]
```

For inspection or feature extraction, `forward_features` returns the complete
encoded sequence:

```python
encoded_tokens = model.forward_features(images)
```

with shape:

```text
[B, N + 1, D]
```

## 13. Training Flow

The separate training module connects CIFAR-10 to the model:

```text
CIFAR-10 training images and labels
        ↓
DataLoader creates shuffled batches
        ↓
Normalize images
        ↓
VisionTransformer(images)
        ↓
Class logits [B, 10]
        ↓
CrossEntropyLoss(logits, labels)
        ↓
Backpropagation
        ↓
Adam updates learnable parameters
```

The original 50,000-image training set is divided into 45,000 training and
5,000 validation images. After each epoch, validation accuracy is measured
without gradient calculation. The checkpoint with the best validation
accuracy is saved.

The original 10,000-image test set is never used to select model weights.
After training finishes, the best validation checkpoint is loaded and the
test set is evaluated once. This prevents test-set leakage into model
selection.

Training arguments are supplied from the command line, so model dimensions
and training settings can change without editing the source code:

```bash
python -m training.train --patch-size 4 --num-layers 6 --batch-size 128
```
