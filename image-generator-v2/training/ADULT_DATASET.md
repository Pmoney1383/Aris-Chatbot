# Verified-adult training data

The training pipeline includes `dataset/nsfw-images` only through
`dataset/nsfw-images/manifest.csv`. It does not discover images automatically
or infer age, consent, captions, or provenance from file and directory names.

The manifest must contain these columns:

```csv
file_path,caption,width,height,age_verified_adult,consent_verified,rights_verified
```

- `file_path` is relative to `dataset/nsfw-images` (a path beginning with
  `dataset/nsfw-images/` is also accepted).
- All three verification fields must be `true`, `yes`, or `1`. A row without
  every attestation is excluded before preprocessing.
- `width` and `height` are positive integer pixel dimensions.
- Do not include ambiguous-age material. Keep the evidence supporting these
  attestations outside the model-training repository.

Every accepted caption receives the stable prefix
`content rating: adult; verified adults only.`. Phase 1 and Phase 2 then draw
the configured `adult_sample_fraction` (5% by default) from the prepared adult
subset. The training command fails instead of silently continuing when that
fraction is enabled but no adult samples have made it through preprocessing.

After adding or changing the manifest, rerun both incremental stages before
training:

```powershell
../.venv/Scripts/python.exe preprocess.py text
../.venv/Scripts/python.exe preprocess.py latents --res 256
```

Run the latent stage again with `--res 512` before Phase 2. Training on marked
adult examples makes content conditioning more separable; an inference-time
prompt policy and safety classifier are still required to enforce refusal or
blocking behavior.
