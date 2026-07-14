# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Aris is a from-scratch LLM training project. Stage 0 (current): pretrain a 124M GPT-2-style decoder-only transformer on FineWeb-Edu (sample-10BT, ~1.5B token budget) on a single RTX 5080 (16GB, CUDA 12.8). Later stages (see `TRAINING_PLAN.md`) add architectural upgrades (RoPE, RMSNorm, SwiGLU), a custom tokenizer, and SFT on the user's personal iMessage logs to turn the base model into a chatbot.

All Python runs through the local venv: `.venv/Scripts/python.exe` (Python 3.14, torch cu128). Torch must be installed from the cu128 index (see `requirements.txt` header) — plain `pip install torch` gives the CPU wheel.

## Commands

- `.venv/Scripts/python.exe scripts/prepare_data.py` — streams FineWeb-Edu, tokenizes with tiktoken gpt2, writes uint16 shards to `data/shards/` (resumable; skips complete shards).
- `.venv/Scripts/python.exe scripts/train.py` — trains from shards; auto-resumes from the latest checkpoint in `checkpoints/`; logs to `logs/loss_log.csv`; prints sample continuations at the end. Ctrl+C saves a checkpoint before exiting.
- `.venv/Scripts/python.exe scripts/chat.py` — text-continuation REPL against the latest checkpoint. This is a base model, not a chatbot (no chat behavior until SFT).
- `.venv/Scripts/python.exe scripts/message_cleaner.py` — cleans a raw iMessage export from `data/raw/` into `data/raw/clean_tagged.txt` (edit `INPUT_FILE` first). Not part of Stage 0 training; used in the SFT stage later.

There is no test suite; verification is done by running the scripts (a smoke test pattern: instantiate the model, check ~124M params, forward/backward one batch).

## Architecture

- **`src/config.py` is the single source of truth for hyperparameters** (`ModelConfig`, `TrainConfig`, `GenerationConfig`). Never hardcode a hyperparameter in a script — add it to config and import it. Checkpoints store `model_config` so `chat.py` reconstructs the exact architecture from the checkpoint, not from the current config.
- **`src/model.py`** — hand-written GPT-2 style transformer: manual attention blocks using `F.scaled_dot_product_attention(is_causal=True)` (FlashAttention on CUDA), pre-norm LayerNorm, GELU FFN, learned positional embeddings, tied input/output embeddings, GPT-2 init (std 0.02, residual projections scaled `1/sqrt(2*n_layers)` via the `IS_RESIDUAL_PROJ` flag). `forward(input_ids, targets=None)` returns `(logits, loss)`.
- **`src/dataset.py`** — `ShardedDataset` memmaps `shard_*.bin` files (flat uint16 token ids) and yields non-overlapping `(input, target)` windows shifted by one token; windows don't cross shard boundaries. `__getstate__` drops memmap handles so `DataLoader(num_workers>0)` works. Train/val split is by shard (last shards = val).
- **`scripts/train.py`** — bf16 autocast, `torch.compile` with eager fallback, AdamW, manual cosine LR schedule with linear warmup, gradient accumulation (`grad_accum_steps`), grad clipping. Checkpoints (`ckpt_{step:06d}.pt`) contain model + optimizer state + step + config, saved every `save_every` steps and on exit/interrupt; resume is automatic from the newest one.

## Repo layout notes

- `archive/` holds the entire pre-Stage-0 project (old seq2seq/decoder chatbot trained directly on iMessage logs: `main_old.py`, `model_old.py`, `chat_old.py`, unused preprocess pipelines, `*.pkl`, `- Copy` backups). Reference only — don't wire new code into it.
- `data/` is gitignored. `data/raw/` contains the user's real personal message history (`clean_tagged.txt`, phone-number-named exports) — treat as sensitive, never print full contents or commit it.
- `checkpoints/`, `logs/`, `data/shards/` are generated artifacts, gitignored, and created by the scripts themselves.
