# TRAINING_PLAN.md — From-Scratch LLM (Aris → 1B)

Plan for training a decoder-only transformer **from scratch** (no pretrained weights), then
fine-tuning it into a chatbot. Target: **≥1B parameters**, trained on an RTX 5080 (16GB).

This is the successor to the current Aris project (word-level, iMessage-only). Everything
here is a scale-up + modernization of that same core idea: continuous token stream +
sliding-window next-token prediction.

---

## Goal

- Train a GPT-2/GPT-3-style base model **ourselves**, from random init.
- Then fine-tune it on our own conversational data (no fine-tuning a pretrained OSS model).
- Reach at least 1B params — but note: **the pipeline and clean SFT stage matter more than
  hitting exactly 1B.** A well-trained 500M can beat a starved 1B.

---

## Hardware reality (RTX 5080, 16GB) — this shapes everything

1B params, training with Adam:

| Component | Memory |
|---|---|
| Weights (bf16) | ~2 GB |
| Gradients (bf16) | ~2 GB |
| Adam states (fp32 m+v) | ~8 GB |
| Activations + CUDA overhead | remainder (~3-4 GB) |

**1B fits only with:** 8-bit Adam (bitsandbytes), gradient checkpointing, bf16 autocast,
`torch.compile`, FlashAttention (PyTorch SDPA), small micro-batch (4-8 @ 1024 ctx) +
gradient accumulation to an effective batch of ~0.5M tokens.

**Throughput:** ~25-40k tok/s for 1B w/ checkpointing → ~2-3B tokens/day best case.
Chinchilla-optimal for 1B ≈ 20B tokens → **~7-10 days minimum, realistically 2+ weeks**
with restarts. **Checkpoint-resume is mandatory from day one** (no more overwrite-every-epoch
like Aris).

---

## The staged ladder

| Stage | Size | Tokens | Time | Purpose |
|---|---|---|---|---|
| **0** | 124M | ~1.5B | ~4 hrs | *Working model TODAY* — proof of life |
| **A** | 124M | ~2.5B+ | ~2-3 days | Validated pipeline, own tokenizer, full data mix |
| **B** | ~500M | scale test | ~1 week | Confirm memory tricks + throughput hold; go/no-go on 1B |
| **C** | 1B | ~15-20B | ~2 weeks | The main event |
| **SFT** | — | — | ~1-3 days | Conversational fine-tune + optional persona pass |

Stage 0 is **not throwaway** — it's literally the seed of Stage A (same model, same code,
just stopped early). The afternoon's work rolls straight forward.

---

## Stage 0 — "It's alive" run (today, one afternoon)

**Point of this stage: validation + dopamine, not quality.** Prove the full pipeline runs
end-to-end and that loss drops + output is coherent English.

**Model — GPT-2 small clone, ~124M params:**

| Param | Value |
|---|---|
| Layers | 12 |
| d_model | 768 |
| Heads | 12 |
| Context | 1024 |
| Vocab | 50257 (tiktoken gpt2) |
| Params | ~124M |

**Key shortcut for today:** skip training our own tokenizer — use `tiktoken`'s GPT-2 BPE
(`gpt2` encoding, 50257 vocab) off the shelf. Removes an entire build step from the critical
path. We train our own 32k BPE tokenizer later for the real stages.

**VRAM:** non-issue here. 124M with full fp32 AdamW ≈ ~2GB. No checkpointing, no 8-bit Adam,
no tricks — crank batch size and go fast.

**Data:** stream ~1.5B tokens of **FineWeb-Edu**, pre-tokenize to a couple of flat `uint16`
binary shards. Pure FineWeb-Edu is fine for proof-of-life; no diversity mix needed yet.

**Time math:** tight 124M pipeline (bf16, `torch.compile`, SDPA flash attention) ≈ 80-120k
tok/s. At ~100k tok/s, 1.5B tokens ≈ **~4 hours.**

**Expectations (read this before judging output):** at ~1.5B tokens a 124M model is
*undertrained* vs its ~2.5B Chinchilla point, and **there's no fine-tuning yet — it's a
text-continuation model, not a chatbot.** Success looks like: locally coherent, grammatical
English that stays on-topic for a sentence or two then drifts. It will **not** answer
questions or hold a conversation — that's the SFT stage's job. Coherent continuation = win.

**Build order (to be training within the hour):**
1. Tokenize-to-shards script (download + encode FineWeb-Edu → `uint16` shards)
2. Model + train loop with sampling (clean, resumable, carries forward to later stages)

---

## Architecture (real stages — Stage A onward)

Modern Llama-style decoder block, from scratch in PyTorch (~300 lines):

- **RoPE** positional encoding (not the sinusoidal Aris uses — RoPE is standard now + better)
- **RMSNorm** + **SwiGLU** FFN, **pre-norm**
- Causal masking via PyTorch SDPA (gets FlashAttention for free)
- Tied input/output embeddings
- **BPE tokenizer, ~32k vocab**, trained ourselves with HuggingFace `tokenizers` on a corpus
  sample (word-level vocab like Aris does NOT scale)

**1B config (Stage C):**
- ~24 layers, d_model 2048, 16 heads (GQA optional, not required)
- Context 1024 (2048 doubles activation cost — start at 1024, extend later)
- Token budget ~15-20B, cosine LR decay, warmup ~2k steps

---

## Pretraining data (the base model)

Pre-tokenize everything **once** into flat binary shards (uint16 token ids, memmap'd). Same
continuous-stream + sliding-window idea as Aris, at scale. Dataloader stays dead simple.

**Primary: FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) — filtered, deduped Common Crawl
scored for educational quality; best open pretraining corpus per token. Streamable via
`datasets`; pull a ~20B token slice (~50-60GB tokenized).

**Diversity mix (~80/10/10) for Stage A onward:**
- **80%** FineWeb-Edu (general web, knowledge, prose)
- **10%** SlimPajama books + StackExchange slices, or Project Gutenberg (long-form structure,
  fiction dialogue)
- **10%** The Pile OpenSubtitles / Ubuntu IRC or similar (raw conversational rhythm, so chat
  isn't alien to the base model)

**Rejected:** Wikipedia-alone (too narrow), Kaggle scraps (low quality/scale).

---

## Fine-tuning data (making it a chatbot)

After pretraining, ~1-3 days of SFT:

1. **OASST1 + OASST2 (OpenAssistant)** — human-written multi-turn conversations, highest
   quality open chat data. Core of the mix.
2. **WildChat-1M** or **LMSYS-Chat-1M** — real user↔LLM conversations for volume + diversity.
   Filter for quality (English, non-toxic, multi-turn).
3. **Own iMessage corpus (from Aris)** — final light persona pass so it talks like me.
   General base → assistant fine-tune → persona fine-tune.

**Critical detail:** format everything into one chat template
(`<|user|> ... <|assistant|> ... <|eot|>` style — same scheme as Aris, better tokens) and
**mask the loss so we only train on assistant/response tokens, not user turns.** That loss
masking is the main thing separating proper SFT from "more pretraining on chat logs."

---

## Deliverables checklist (build order)

1. [ ] **Stage 0:** tokenize-to-shards script (FineWeb-Edu, ~1.5B tokens, tiktoken gpt2)
2. [ ] **Stage 0:** 124M model + train loop w/ sampling, resumable
3. [ ] **Stage 0:** run ~4 hrs → confirm coherent English continuation
4. [ ] BPE tokenizer training script → `tokenizer.json` (32k vocab)
5. [ ] Full corpus download + tokenize pipeline (streaming, resumable, diversity mix)
6. [ ] Model upgrade: Llama-style block (RoPE, RMSNorm, SwiGLU)
7. [ ] Training loop: bf16, grad accum, grad checkpointing, 8-bit Adam, cosine schedule,
       **checkpoint + resume + wandb/tensorboard logging**
8. [ ] **Stage A** (124M, own tokenizer) → validate loss ~3.0-3.3 on FineWeb-Edu
9. [ ] **Stage B** (~500M) → go/no-go decision on 1B
10. [ ] **Stage C** (1B, ~2 weeks)
11. [ ] SFT pipeline w/ loss masking → assistant model
12. [ ] Optional persona pass on iMessage data

---

## Notes / reminders

- Checkpoint-resume from day one. Version checkpoints (don't overwrite like Aris).
- Hyperparams that define a checkpoint (layers, d_model, heads, ctx, vocab) must match between
  training and loading code — same footgun as Aris `main.py`/`chat.py`.
- The "≥1B" target is the *least* important part of this plan. Pipeline + modern architecture
  from scratch + clean SFT = the resume-worthy substance, whether the final checkpoint is
  500M or 1B.
