# Text-to-Image Experiment Plan

## Goal
Build a private learning/research pipeline for a text-conditioned image generator using a broad image dataset, structured captions, local safety filtering, and latent diffusion.

## Dataset Collection
Use a category catalogue to keep the dataset diverse instead of scraping random images blindly.

Example categories:
- Animals
- People
- Places
- Objects
- Food
- Vehicles
- Architecture
- Nature
- Fashion
- Technology
- Art / illustration
- Adult content
- etc

For each category, generate many search phrases and collect high-quality source images.

Preferred source quality:
- Keep original images.
- Prefer short side >= 1080 px.
- Do not force everything to 1920x1080 or 16:9.
- Resize/crop dynamically during training.

## Cleaning Pipeline

```text
image discovery
    ↓
source/provenance check
    ↓
local download quarantine
    ↓
corrupt/resolution check
    ↓
duplicate detection
    ↓
watermark detection → reject
    ↓
local safety gate (this will come later in a completley different branch)
    ↓
captioning / metadata API
    ↓
final dataset
```

### Watermarks
Detect and reject watermarked or stock-overlay images using pre-trained model such as VGG16

### Duplicates
Use:
- SHA-256 for exact duplicates
- pHash for near-identical images
- Optional CLIP/image embeddings for visually similar images

## Adult Content / Safety

- the goal for this is to later learn model safety guideline and how to enforce it onto AI

Important rules:
- Do not send suspected or age-ambiguous sexual content to cloud AI APIs.
- Run a local safety/provenance gate first.
- avoid anything including a person who is a minor

## Captioning / Metadata
After an image passes local filtering, send it to a cheap vision API for structured annotation.

Example JSON:

```json
{
  "caption": "A blue sports car driving along a coastal highway at sunset.",
  "objects": ["sports car", "road", "ocean", "mountains"],
  "style": "photograph",
  "setting": "coastal highway",
  "lighting": "sunset",
  "quality_score": 0.91,
  "aesthetic_score": 0.84,
  "watermark": false
}
```

Use one image per request/job so failures and retries stay simple.

## Dataset Manifest
Suggested fields:

```text
id
category
subcategory
file_path
source_url
source_domain
sha256
phash
width
height
caption
objects[]
style
setting
lighting
quality_score
adult_label
safety_status
```

## Training Architecture

```text
text prompt
    ↓
pretrained text encoder (CLIP/T5-style)
    ↓
text embeddings
    ↓
latent diffusion U-Net or DiT
    ↓
clean latent
    ↓
pretrained VAE decoder
    ↓
image
```

Use latent diffusion rather than raw-pixel diffusion for efficiency.

### Local Training if possible on a RTX 5080
- Train denoiser in BF16
- Keep VAE frozen
- Keep text encoder frozen
- Quantize frozen text encoder to 8-bit/4-bit if useful
- Use an 8-bit optimizer if needed
- Enable gradient checkpointing
- Use efficient attention
- Use gradient accumulation for small VRAM

Avoid training the main denoiser directly in 4-bit unless specifically experimenting with low-bit training.

## Resolution Strategy
Keep high-resolution originals, but start training smaller.

```text
256px prototype
    ↓
512px main experiment
    ↓
768/1024px only if results justify the cost
```

Use aspect-ratio buckets instead of forcing all images into squares.

## Overall Idea

```text
internet / curated datasets
        ↓
category-balanced collection
        ↓
cleaning + dedup + watermark rejection
        ↓
local safety gate
        ↓
structured captioning
        ↓
image-caption dataset
        ↓
text encoder + latent diffusion + VAE
        ↓
generated image
        ↓
separate safety classifier
```
