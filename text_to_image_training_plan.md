# Text-to-Image Training Plan

## Goal
Train a custom text-to-image model from scratch using ~1,110,404 image-caption pairs, while reusing pretrained components for compression and text understanding.

## Model Target
- **Trainable denoiser:** ~400M parameters
- Practical range: **350M–450M**
- Architecture: **DiT (Diffusion Transformer)**
- Native generation target: **512-class resolution**
- Final high-resolution output can be produced with a pretrained upscaler.

## Core Architecture

```text
caption
  ↓
pretrained frozen text encoder
  ↓
text embedding

image
  ↓
pretrained frozen VAE
  ↓
latent

noisy latent + timestep + text embedding
  ↓
~400M DiT
  ↓
denoised latent
  ↓
VAE decoder
  ↓
image
```

The DiT is the part trained from scratch.

## Preprocessing
Do this once before training:

1. Resize/crop images into aspect-ratio buckets.
2. Encode every image with the frozen VAE.
3. Save the resulting latents to disk.
4. Encode every caption with the frozen text encoder.
5. Save text embeddings to disk.
6. Store data in shards, e.g. 5k–10k samples per shard.

Training should then load only:

```text
latent + text embedding + metadata
```

instead of loading the VAE, text encoder, and original image every step.

## Resolution Strategy

### Phase 1 — 256px
Train the same DiT at lower resolution first.

Purpose:
- learn objects
- learn composition
- learn prompt/image relationships
- learn styles cheaply

### Phase 2 — 512px
Continue from the Phase 1 checkpoint using 512-class aspect-ratio buckets.

Examples:

```text
512x512
576x448
448x576
640x384
384x640
```

Keep roughly similar pixel counts per bucket.

## 16 GB VRAM Optimization

Use:

- **BF16** for the trainable DiT
- pretrained **frozen VAE**
- pretrained **frozen text encoder**
- precomputed VAE latents
- precomputed text embeddings
- **gradient checkpointing**
- **PyTorch SDPA / Flash Attention**
- **8-bit AdamW**
- microbatch **1–2**
- **gradient accumulation** for a larger effective batch
- **EMA on CPU**
- mixed precision
- gradient clipping
- optional `torch.compile()` after the training loop is stable

Avoid 4-bit training for the main DiT initially.

Possible starting point:

```text
microbatch: 1
gradient accumulation: 32–64
effective batch: 32–64
```

Increase only if VRAM allows.

## Training Objective

Keep the first version simple:

```text
clean latent
  ↓
add random noise at timestep t
  ↓
DiT predicts noise / velocity
  ↓
loss against target
```

Use a standard diffusion or velocity-prediction objective.

Optional later improvement:
- Min-SNR loss weighting
- flow matching
- FP8 transformer operations

Do not add these until the basic model trains correctly.

## Optimizer / Stability

Suggested starting values:

```text
optimizer: AdamW8bit
learning rate: ~1e-4
weight decay: ~0.01
gradient clipping: 1.0
precision: BF16
EMA: yes
```

Use a warmup and then cosine decay or another simple scheduler.

## Checkpoints & Evaluation

Save checkpoints regularly:

```text
every 2k–5k optimizer steps
```

Generate a fixed validation prompt set at each checkpoint.

Keep the same:
- prompts
- seeds
- sampler
- inference steps

This makes visual progress easy to compare.

Watch for:
- improving prompt adherence
- anatomy
- object quality
- composition
- diversity
- overfitting / memorization
- later checkpoints becoming worse than earlier ones

Keep the best checkpoint, not automatically the latest one.

## Dataset Sampling
Shuffle globally rather than exhausting categories in order.

Try to avoid large category imbalance.

If some categories dominate the 1.11M images, use weighted sampling so rare concepts are still seen often enough.

## Safety
Keep the safety system separate from the generator initially.

```text
prompt
  ↓
prompt safety classifier
  ↓
generator
  ↓
image safety classifier
  ↓
allow / block / blur / uncertain
```

This makes it easier to measure what the raw generator learned versus what the safety layer blocks.

## Suggested Development Order

```text
1. preprocess a small test subset
2. verify VAE latent pipeline
3. verify text embeddings
4. build ~400M DiT
5. overfit a tiny subset as a sanity test
6. train at 256px
7. inspect samples/checkpoints
8. continue at 512px
9. tune optimizer/loss only if needed
10. add safety layer
11. add pretrained upscaler
```

## Final Target

```text
Dataset: ~1,110,404 image-caption pairs
Trainable model: ~400M DiT
VAE: pretrained + frozen
Text encoder: pretrained + frozen
Phase 1: 256px
Phase 2: 512-class aspect buckets
Training precision: BF16
Optimizer: AdamW8bit
VRAM target: 16 GB
Native output: ~512px
Final output: upscale to 1K / 2K
```
