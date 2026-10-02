# Text-to-Image Training Plan

_Last updated: 2026-09-30. Code lives in `image-generator-v2/training/`; all hyperparameters are in `training/config.py`._

## Goal
Train a text-to-image model from scratch that produces **solid, recognizable images** — "not bad for a local model" — on a single RTX 5080 (16 GB VRAM, 31 GB RAM), reusing pretrained components for compression and text understanding.

## Architecture (built)

```text
caption ──► Flan-T5-large encoder (frozen, 1024-d, ≤160 tokens) ──► text tokens
image ───► SDXL VAE (frozen, sdxl-vae-fp16-fix, 4 ch, /8) ───────► latent

noisy latent + timestep + text tokens
        ▼
DiT denoiser — 415.0M params, trained from scratch
  hidden 1024 · depth 24 · 16 heads · patch 2
  cross-attention to T5 tokens · adaLN-single (timestep + pooled caption)
  2D RoPE (works for every aspect bucket and across resolutions) · QK-norm
        ▼
velocity ──► Euler sampler (30 steps, CFG 4.5) ──► latent ──► VAE decoder ──► image
```

- **Objective:** rectified flow. `x_t = (1 - t)·x0 + t·noise`, target `v = noise - x0`, MSE loss; timesteps ~ logit-normal(0, 1).
- **Classifier-free guidance:** 10% caption dropout during training; sampling at **CFG 4.5, 30 steps** (a CFG 1/2/3/4.5 sweep and 30-vs-50 steps confirmed this).
- **Aspect buckets:** 11 buckets with ~equal pixel counts (e.g. 256×256, 288×224, 320×192, 352×176 and portrait mirrors at 256-class; ×2 at 512-class).

## Data

| Set | Images | Captions | Status |
|---|---|---|---|
| Gathered (PD12M + Megalith) | 916,743 | real captions | ✅ preprocessed at 256px |
| Gathered, uncaptioned (Pixabay/Pexels/iNat/Wikimedia) | ~193k | folder labels only | ⏳ needs captions |
| Phase 1.5 concepts | ~110k target (~10.8k collected) | labeled (car model, dish, species, scene) | ⏳ collector paused (RAM) |

- Phase 1.5 concepts (`scripts/phase15_concepts.py`): 402 concepts — 91 car models, 166 dishes (incl. Persian and many cuisines), 15 drinks, 88 animal species, 15 city/night/weather scenes, 11 space, 16 tech — all ≥512 px, from Wikimedia Commons categories, iNaturalist (research-grade, wild) and NASA.
- Preprocessing (`training/preprocess.py`) is incremental: `text` (T5), `latents --res N` (VAE per bucket), `flips --res N` (mirrored latents), `verify`.
- **Black-and-white images:** ~15% of PD12M and ~5% of Megalith images are grayscale, and ~40% of those captions don't say so — a cause of prompts drifting to B&W.

## Phases

| Phase | Resolution | Steps | Samples seen | Starts from | Time (5080) | Status |
|---|---|---|---|---|---|---|
| **0** overfit test | 256 | 1,500 | — | scratch | 15 min | ✅ passed (reproduces its 64 training images) |
| **1** base | 256 | 40,000 | ~10M | scratch | ~2.5 days | 🔄 running — val loss 0.816 → 0.7655 @ 27.5k, finishes ~Thu 2026-10-01 04:00 |
| **1.5** extended + concepts | 256 | **~120,000** | **+~31M (total ~41M)** | phase 1 best | **~6 days** | ⏳ next |
| **2** detail | 512 | 20,000 | +~5M | phase 1.5 best | ~4.5 days (or ~11–18 h on an H200) | later |

Effective batch 256 throughout (256px: 64 × 4 accumulation; 512px: 16 × 16).

### Why Phase 1.5 is long
The model is **under-trained, not broken**. Checked and ruled out: pipeline bugs (overfit test), VAE (27 dB reconstructions), objective/sampler, CFG, EMA lag (EMA vs raw weights ~4.6% apart, same images). What limits quality is **samples seen**: ~10M after phase 1, versus the ~50–100M typical for crisp objects at this model size. Phase 1.5 takes the total to ~41M.

### Phase 1.5 changes (apply only after phase 1 finishes — never edit training code during a run)
1. `TRAIN_PHASES["1.5"].total_steps` 35k → **~120k**; peak LR 6e-5, 1k warmup, cosine decay.
2. **Data:** general set + Phase 1.5 concepts, mixed via category-balanced sampling.
3. **Multi-length captions:** each image gets detailed / medium / short-label captions (e.g. "Ferrari 488"), sampled at random (~60% detailed, 20% medium, 15% short, + caption dropout) so short prompts like "A dog." work.
4. **B&W tagging:** prepend "Black-and-white photograph." to captions of images detected as grayscale from pixel saturation.
5. **Flip augmentation** (`hflip=True`): pre-encoded mirrored latents used 50% of the time; captions mentioning left/right are never flipped. (Flipping latents directly is wrong — the SDXL VAE isn't mirror-symmetric: 18.6 vs 26.5 dB.)
6. **CLIP-score evaluation** at every checkpoint (~50 fixed descriptive prompts × 4 seeds). If it stops improving well before 120k, stop early. Keep a checkpoint every ~20k steps for comparison.

## Order of work after Phase 1

1. Descriptive-prompt test at step 40k (compare with the 20k and 27.5k tests).
2. Finish Phase 1.5 collection: `scripts/gather_images.py --phase15 --no-megalith` (~7 h; only while training is stopped — RAM).
3. Captioning script (Florence-2-large): `<DETAILED_CAPTION>` + label prefix ("Ferrari 488, a red sports car…"), plus medium/short variants and B&W tags; also caption the ~193k uncaptioned gathered images.
4. Export captions → `preprocess.py text` → `latents --res 256` → `flips --res 256`.
5. `train.py --phase 1.5` (~6 days), tracking CLIP score.
6. Phase 2 at 512px: `latents --res 512` + `flips --res 512` (~6 h), then `train.py --phase 2` — locally or on an H200.

## Training setup (16 GB VRAM)

- BF16 autocast, fp32 master weights, 8-bit AdamW (bitsandbytes), gradient clipping 1.0, weight decay 0.01.
- Gradient checkpointing on (without it even micro-batch 32 doesn't fit next to the rest); peak ~5.2 GB at 256px batch 64, ~5.1 GB at 512px batch 16.
- EMA (decay 0.9999, updated every 10 steps) kept on the CPU; checkpoints every 2,500 steps (last 5 kept + `best.pt` by validation loss); deterministic validation loss on ~2k held-out images; sample grid of 8 short prompts per checkpoint.
- Live status line, safe Ctrl+C (saves and resumes at the exact batch), auto-resume.
- RAM is the tight resource: training commits ~26 GB (4 loader workers). Don't run the collector or other heavy jobs alongside it.

## Evaluating quality

- Judge with **descriptive prompts** written like the training captions ("A golden retriever sitting on green grass in a park…"), several seeds each. The 8 short validation prompts ("A dog.") badly undersell the model, and single seeds swing a lot between checkpoints.
- Track: validation loss, CLIP score, and fixed-seed comparisons across checkpoints.

## Safety
Keep safety separate from the generator:

```text
prompt ► prompt safety classifier ► generator ► image safety classifier ► allow / block / blur
```

Concept-erasure experiments (ESD/UCE, negative guidance) on an already-trained checkpoint, measured for leakage and collateral damage, come after the base model is solid.

## Later
- Pretrained upscaler for 1K/2K output.
- Scale-up options: ~3–5M images and/or a ~0.6B DiT, trained toward 50–100M samples (an H200 makes this ~2–3 days instead of weeks).
