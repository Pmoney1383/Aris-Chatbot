# DIFFUSION_PLAN.md — From-Scratch DDPM (Aris Vision, Stage D0)

Plan for building a from-scratch diffusion model that generates cat images on an
RTX 5080 (16GB), running in WSL2. Same philosophy as the Aris LLM project and the
CatGAN project: train it ourselves, no pretrained weights, staged approach, learn
the fundamentals — and this time, no adversarial training to fight.

You said quality matters more than time, so this plan is sized to actually push
64×64 quality as far as a from-scratch model reasonably can on this hardware,
not to finish fast. Expect a genuinely long run (see the time budget section) —
that's intentional.

This lives alongside `vision/` (the GAN project). New subfolder so both projects
coexist cleanly.

---

## Why diffusion instead of more GAN tuning

You've already been through: DCGAN baseline instability, lr tuning, spectral
norm, R1 gradient penalty, self-attention, minibatch stddev — and still hit a
slow structural drift toward D dominance over a full 100k-step run. That's not
a failure, it's a real result: plain adversarial training has a ceiling here,
and you've now seen exactly where it is.

Diffusion sidesteps the entire adversarial dynamic. There's no discriminator,
no D/G balance to fight, no mode collapse. Training target is direct and
simple:

```python
noise = torch.randn_like(images)
noisy_images = add_noise(images, noise, timesteps)
predicted_noise = model(noisy_images, timesteps)
loss = F.mse_loss(predicted_noise, noise)
```

That's the entire training objective. No two-network balancing act. The
tradeoff is generation speed (many denoising steps to sample one image) and
usually more total compute to reach comparable quality — which you've said
is fine.

---

## Project ladder

| Stage | Project | Resolution | Time budget | Purpose |
|---|---|---|---|---|
| **D0** | DDPM cats (this plan) | 64×64 | ~24-48 hrs | Best achievable native 64×64 quality, no pretrained weights |
| **D1** | Upscale pipeline | 64→256 | ~1 hr setup | Real-ESRGAN on top of D0 outputs (same as G1 in the GAN plan) |
| **D2** (optional, later) | Fine-tune pretrained diffusion w/ LoRA | higher native res | varies | Only after D0 is understood and you want to go past from-scratch quality |

D0 is sized generously — bigger U-Net, longer training, more sampling steps —
specifically because you said you don't mind the wait and want the best 64×64
result achievable, not the fastest one.

---

## How a DDPM works (the short version)

**Forward process (fixed, no learning):** gradually add Gaussian noise to a
real image over T timesteps (T=1000 is standard) until it's pure noise. This
is defined by a noise schedule (a sequence of beta values controlling how much
noise gets added at each step) — it's math, not a trained component.

**Reverse process (this is what the model learns):** a U-Net is trained to
predict the noise that was added at a given timestep, given the noisy image
and the timestep itself. Once trained, generation works by starting from pure
random noise and repeatedly asking the U-Net "what noise is in this?", then
subtracting a bit of it — repeated across T steps (or fewer, with DDIM
sampling) until a clean image emerges.

Why it's more stable than GANs: the loss is a direct, well-behaved regression
target (MSE between predicted and actual noise) rather than a moving
adversarial target. There's no equilibrium to maintain between two competing
networks — just one network with a clear, consistent job.

---

## Architecture (sized for quality, not speed)

This is deliberately larger than a "toy" DDPM tutorial model, since you want
strong 64×64 quality and don't mind the training time.

**U-Net backbone:**
- Base channels: 128 (up from the "minimal" 64 — more capacity for detail)
- Channel multipliers: `[1, 2, 2, 4]` → resolutions 64→32→16→8, channel widths
  128→256→256→512
- Residual blocks per resolution: 3 (up from the typical minimal 2, for more
  representational depth)
- Normalization: GroupNorm (standard for diffusion U-Nets, not BatchNorm —
  works well at batch size 1 too, matters less here but it's the correct
  default)
- Activation: SiLU (swish) throughout — standard in DDPM/ADM-style U-Nets
- Timestep conditioning: sinusoidal timestep embedding (like transformer
  positional embeddings) → small MLP → added into every residual block via
  FiLM-style scale/shift, not concatenation
- **Attention placement (important — this is the exact mistake flagged from
  the GAN's self-attention slowdown):** self-attention ONLY at 16×16 and 8×8
  internal resolutions. NEVER at 64×64 or 32×32 — full self-attention at 64×64
  is a 4096×4096 attention matrix per layer, which is exactly what caused the
  GAN to go from 30 minutes to 4 hours. At 16×16 that's 256×256 — trivial. At
  8×8 it's 64×64 — free.
- Skip connections: standard U-Net encoder→decoder skip connections at each
  resolution level

Estimated params: roughly 60-90M depending on exact channel counts — meaningfully
bigger than the GAN's G+D combined (~5.5M), but well within a 16GB card for
diffusion training given gradient checkpointing.

**Noise schedule:**
- T = 1000 timesteps
- Schedule: cosine (Nichol & Dhariwal, "Improved DDPM") rather than the
  original linear schedule — cosine preserves more signal in the middle
  timesteps and consistently produces better sample quality than linear at
  small/medium scale. Worth the extra ~10 lines of code over linear.

**Sampling:**
- Training doesn't touch this — one random timestep per training image, one
  U-Net forward/backward pass, same cost profile as any other network.
- For generation: implement both full 1000-step DDPM ancestral sampling (best
  quality, slowest) and DDIM sampling (20-50 steps, faster, negligible quality
  loss) — use DDIM for all the frequent progress-check grids during training,
  save full DDPM sampling for final "best checkpoint" output.

---

## Training config

```python
@dataclass
class DiffusionConfig:
    # data
    image_size: int = 64
    channels: int = 3
    dataset_name: str = "huggan/cats"   # reuse vision/data/processed/cats_64.pt

    # model
    base_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 4)
    num_res_blocks: int = 3
    attn_resolutions: tuple = (16, 8)     # NEVER include 64 or 32 here
    dropout: float = 0.1                   # diffusion U-Nets benefit from some dropout, unlike the GAN

    # noise schedule
    timesteps: int = 1000
    schedule: str = "cosine"

    # training
    batch_size: int = 48        # start here, back off if VRAM overflows (see below)
    lr: float = 2e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    ema_decay: float = 0.9999    # EMA of model weights for sampling — standard
                                    # practice, sampling from the EMA model is
                                    # meaningfully better than the raw weights
    grad_clip: float = 1.0
    max_steps: int = 300_000     # generous — quality over time budget
    sample_every: int = 2000     # DDIM 50-step samples, fixed noise seed like the GAN
    ckpt_every: int = 5000
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
```

**Batch size note:** 48 is a starting estimate, not a guarantee — diffusion
U-Nets are memory-heavier per sample than the GAN's generator was. If you hit
VRAM overflow, drop to 32, then 24. Gradient checkpointing is on by default to
help afford a larger batch at this model size. Because you said time doesn't
matter, prioritize **stability over squeezing batch size up** — a smaller
batch that trains cleanly beats a larger one that OOMs.

**Total images seen matters more than step count** when comparing to the GAN
run — worth logging `images_seen = step * batch_size` alongside step count in
your training log this time, so any future comparison (including if you circle
back to WGAN-GP or another GAN variant) is apples-to-apples.

---

## Time budget (generous, quality-first)

Rough estimates for this sized model at batch ~48, on the 5080, with attention
correctly restricted to 16×16/8×8:

- ~300k steps at this size: realistically **24-48 hours**, possibly more
  depending on how gradient checkpointing trades off against step time
- You should see vague structure emerging much earlier than that (within the
  first few hours) — diffusion models tend to show *rough* shape/color fairly
  early, then keep refining texture/detail steadily for a long time after,
  unlike GANs which can plateau hard. Don't judge this at the "few hours"
  mark the way you might've wanted to judge the GAN at 5k steps — give it
  proportionally more patience since the whole point of this plan is letting
  it run long.
- Sampling (generation) after training: DDIM at 50 steps takes a small
  fraction of a second per image on this GPU — fine for frequent checks. Full
  1000-step ancestral sampling is much slower per-image; only use it for final
  output grids, not every checkpoint.

If it's clearly still improving at 300k, extending further is completely
reasonable given your stated priority — there's no hard ceiling here like
there was with the GAN's structural D-dominance issue. Diffusion models are
well known to keep improving with more training more reliably than GANs do.

---

## Directory structure

```
diffusion/
├── data/                    # symlink or copy of vision/data/processed/cats_64.pt
├── checkpoints/             # model + EMA weights + optimizer state
├── samples/                 # DDIM progress grids per checkpoint
├── src/
│   ├── unet.py               # U-Net with correct attention placement
│   ├── schedule.py           # cosine noise schedule, forward process math
│   ├── config.py             # DiffusionConfig dataclass above
│   └── sampling.py           # DDPM + DDIM samplers
└── scripts/
    ├── train_diffusion.py    # main training loop
    └── generate.py           # sample N images from a checkpoint (DDIM default, --full for 1000-step)
```

---

## CLAUDE CODE PROMPTS

### Prompt 1 — Project setup + U-Net + noise schedule (run first)

---

We're adding a from-scratch diffusion model (DDPM) project to this repo,
alongside the existing `vision/` GAN project. Create a `diffusion/` subfolder
with this structure: `diffusion/data/`, `diffusion/checkpoints/`,
`diffusion/samples/`, `diffusion/src/`, `diffusion/scripts/`.

This trains on the same cat dataset as the GAN (`vision/data/processed/cats_64.pt`,
shape ~[19994, 3, 64, 64], normalized to [-1, 1]) — symlink or reference it
directly from `diffusion/data/`, don't duplicate the ~800MB file.

The Python venv already exists at .venv with torch+cu128 installed.

#### diffusion/src/config.py

```python
@dataclass
class DiffusionConfig:
    image_size: int = 64
    channels: int = 3

    base_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 4)
    num_res_blocks: int = 3
    attn_resolutions: tuple = (16, 8)
    dropout: float = 0.1

    timesteps: int = 1000
    schedule: str = "cosine"

    batch_size: int = 48
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    max_steps: int = 300_000
    sample_every: int = 2000
    ckpt_every: int = 5000
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
```

Plus paths (data dir, checkpoint dir, samples dir) using pathlib relative to
`diffusion/`.

#### diffusion/src/schedule.py

Implement the cosine noise schedule from Nichol & Dhariwal "Improved
Denoising Diffusion Probabilistic Models":

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

From betas, precompute and store as buffers: `alphas`, `alphas_cumprod`,
`sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, and whatever else the
forward process (`q_sample`) and reverse process math need. Write a
`GaussianDiffusion` class that wraps a U-Net (passed in) and exposes:
- `q_sample(x_0, t, noise)` — forward process, adds noise at timestep t
- `p_losses(x_0, t)` — samples noise, calls q_sample, runs the U-Net, returns
  MSE loss between predicted and actual noise
- Leave sampling methods as stubs for now — implemented in Prompt 2's
  `sampling.py`

Add a `if __name__ == "__main__"` block that instantiates the schedule, prints
the beta/alpha values at t=0, t=500, t=999, and confirms they're monotonic and
within expected ranges (betas increasing, alphas_cumprod decreasing toward 0).

#### diffusion/src/unet.py

Build the U-Net from scratch, clearly structured:

- **Timestep embedding:** sinusoidal embedding (same construction as
  transformer positional embeddings, but embedding the integer timestep
  instead of sequence position) → 2-layer MLP with SiLU → produces a
  per-timestep embedding vector, passed into every residual block.
- **ResBlock:** GroupNorm → SiLU → Conv2d → (inject timestep embedding via
  a linear projection added as a per-channel bias/scale, FiLM-style) →
  GroupNorm → SiLU → Dropout → Conv2d, with a residual skip (1x1 conv if
  channel count changes).
- **AttentionBlock:** standard self-attention over spatial positions
  (flatten H×W into a sequence, QKV via 1x1 convs, standard scaled dot-product
  attention, project back). **Only instantiate this block at resolutions
  matching `config.attn_resolutions` (16 and 8) — do not add attention at 64
  or 32.** This is a deliberate constraint from lessons learned on this
  project's GAN — full self-attention at 64×64 caused an 8x training slowdown
  there (30min→4hr) because of the 4096×4096 attention matrix; keep attention
  restricted to the low-res bottleneck stages here.
- **Downsample/Upsample:** strided Conv2d for downsampling, nearest-neighbor
  upsample + Conv2d for upsampling (avoid transposed conv checkerboard
  artifacts — this was also a known issue in the GAN's generator).
- **Overall structure:** standard U-Net — encoder (ResBlock × num_res_blocks
  at each resolution, then downsample) → bottleneck (ResBlock + Attention +
  ResBlock) → decoder (mirror the encoder, with skip connections
  concatenated from corresponding encoder resolution) → final GroupNorm +
  SiLU + Conv2d to 3 output channels (predicting noise, so no final
  activation like Tanh).
- Weight init: standard (default PyTorch init is fine for this architecture,
  don't need the DCGAN-style normal(0,0.02) init here).

Add a `if __name__ == "__main__"` self-test: instantiate the U-Net with the
config, create a dummy batch (batch=4, 3, 64, 64) and dummy timesteps
(batch=4, random ints in [0,1000)), run a forward pass, confirm output shape
matches input shape (4, 3, 64, 64), print total param count.

#### Verification (this prompt only)

Run both self-tests (`schedule.py` and `unet.py`). Then write and run a
tiny standalone check: instantiate `GaussianDiffusion` wrapping the U-Net,
run `p_losses` on a dummy batch (real random tensor standing in for actual
data, since `cats_64.pt` symlink may not be set up yet), confirm the loss is
a finite positive scalar and `.backward()` runs without error. Do NOT start
real training and do NOT run any data preprocessing — report back param
counts and confirm all three checks pass.

---

### Prompt 2 — Sampling + training loop (run after Prompt 1 verified)

---

The U-Net and noise schedule passed their checks. Now:

#### diffusion/src/sampling.py

Implement two samplers, both taking a trained `GaussianDiffusion` + U-Net and
a batch size:

1. **`ddpm_sample`** — full ancestral sampling across all T=1000 timesteps,
   the standard (slow, highest quality) DDPM reverse process.
2. **`ddim_sample`** — DDIM sampling with a configurable number of steps
   (default 50), the deterministic accelerated sampler from Song et al.
   "Denoising Diffusion Implicit Models." Use this as the default for all
   progress-check grids during training since it's dramatically faster.

Both should support a fixed noise seed argument, so we can generate the same
"anchor" grid every checkpoint for visual progress comparison — same pattern
as the GAN's fixed-noise sample grid.

#### diffusion/scripts/train_diffusion.py

- Symlink/load `cats_64.pt` from `vision/data/processed/`, wrap in
  TensorDataset + DataLoader (shuffle, drop_last)
- bf16 autocast, gradient checkpointing enabled on the U-Net per config
- Maintain an EMA copy of the model weights (decay=0.9999), updated every
  step — sampling during training and for final output should use the EMA
  weights, not the raw training weights (standard diffusion practice, gives
  meaningfully cleaner samples)
- Each step: sample random timesteps for the batch, call `p_losses`,
  backward, gradient clip (per config), optimizer step, EMA update
- Track with rich: step, loss (should be noisy but trend down and roughly
  plateau — NOT oscillate around an equilibrium like the GAN did, this is
  a real regression loss), images_seen (step × batch_size), steps/sec, ETA,
  current VRAM usage
- Every `sample_every` steps: generate a fixed grid (64 images from fixed
  noise seed) using `ddim_sample` at 50 steps, save to
  `diffusion/samples/step_{step:06d}.png`
- Every `ckpt_every` steps: save U-Net weights, EMA weights, optimizer state,
  step number to `diffusion/checkpoints/diff_{step:06d}.pt`. Auto-resume from
  latest checkpoint on startup (same pattern as the GAN and LLM projects)
- Handle KeyboardInterrupt: save checkpoint before exit
- Log step, loss, images_seen, steps/sec to `diffusion/logs/diffusion_loss.csv`

#### diffusion/scripts/generate.py

- Loads latest (or --ckpt specified) checkpoint, uses EMA weights
- `--sampler ddim` (default, --steps flag, default 50) or `--sampler ddpm`
  (full 1000-step) 
- Generates N images (default 16, --n flag) from fresh random noise
- Saves individual PNGs plus one grid image to `diffusion/samples/generated/`

#### Verification

Run a short smoke test: 20 training steps on either real data (if the
symlink to `cats_64.pt` is set up) or a random stand-in tensor at the real
batch size, confirm no NaN, confirm loss is a reasonable finite value, confirm
VRAM usage fits comfortably in 16GB, confirm one `ddim_sample` call at the
end produces a valid (4, 3, 64, 64) image batch without erroring. Report
steps/sec and VRAM usage from the smoke test so we can sanity check the
batch_size=48 assumption before committing to a long run. Do NOT start the
real 300k-step training run — report back first.

---

## What to watch during the real run

- **Loss should trend down and roughly plateau over time** — unlike the
  GAN's oscillating equilibrium, this is closer to a normal regression loss.
  Steady, if noisy, downward trend is healthy. A loss that's flat from step 1
  (never decreasing) suggests something is broken (check gradient flow,
  learning rate, or a bug in the noise schedule).
- **Sample grids will look very different in character from the GAN's
  progression** — expect early grids (first few thousand steps) to look like
  coherent blobs of correct-ish color distribution rather than the GAN's
  noisy static, since diffusion is regressing toward a target from the start
  rather than fighting an adversary. Structure and detail should then improve
  steadily and much more predictably than the GAN did — no risk of the
  "collapse to one blob" failure mode you saw with the GAN, since there's no
  generator trying to find a cheap trick against a discriminator.
- **VRAM usage** — watch this closely for the first several hundred steps
  especially, since 48 is an estimate, not a tested number for this exact
  architecture. Back off batch size early if needed rather than risking a
  crash deep into a long run.
- **Don't compare step-for-step against the GAN's timeline.** The GAN's
  "5k = blobs, 15k = color/shape, 30-50k = recognizable cats" schedule
  doesn't transfer — diffusion has a different, generally slower-feeling
  early trajectory but much more reliable steady improvement. Judge this run
  on its own curve.

## After D0 finishes

1. Compare best-checkpoint output quality directly against the GAN's final
   93.5k grid — same dataset, same resolution, real apples-to-apples
   comparison of the two approaches you've now actually built yourself.
2. Run the Real-ESRGAN upscale pass (D1) on whichever model — GAN or
   diffusion — produced the better native 64×64 base, same pipeline as the
   GAN plan's G1 stage.
3. Only after that: decide whether D2 (LoRA fine-tuning a pretrained
   diffusion model) is worth doing, now that the U-Net, noise schedule, and
   sampling process are things you understand from having built them
   yourself rather than treating as a black box.
