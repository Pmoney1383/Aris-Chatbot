# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Aris is a small decoder-only transformer chatbot (PyTorch) trained on personal iMessage-style chat logs. It is a learning project, not a production system — there is no test suite, no build step, and no package manifest (dependencies are installed ad hoc: `torch`, `torchinfo`, `rich`, `matplotlib`, `pandas`, `numpy`).

The model has evolved from an old seq2seq/DailyDialog design (see commit history) to the current setup: a GPT-style decoder-only transformer trained on a continuous token stream of the user's own messages.

## Commands

There is no build/lint/test tooling. The workflow is entirely script-driven:

- `python main.py` — trains the model end-to-end: loads `clean_tagged.txt` via `message_preprocess.py`, builds the vocab, trains `DecoderOnlyTransformer`, saves `chatbot_model.pt` and `vocab.pkl`, and writes `loss_curve.png` / `accuracy_curve.png`.
- `python chat.py` — loads `vocab.pkl` + `chatbot_model.pt` and starts an interactive REPL chat loop (type `exit` to quit).
- `python message_cleaner.py` — one-off ETL: reads a raw exported chat log (`data/+<phone>.txt`), strips timestamps/junk/media placeholders, tags each line `<me>`/`<other>`, merges consecutive same-speaker lines, and writes `clean_tagged.txt`. Edit `INPUT_FILE` at the top before running.
- `python message_preprocess.py` — standalone smoke test of the stream-loading/vocab-building pipeline used by `main.py` (prints token counts, doesn't train anything).

`create_dataset.py` and `custom_preprocess.py` / `preprocess.py` are earlier, now-unused pipelines (custom greeting pairs dataset, and DailyDialog CSV pairs dataset respectively) kept around from prior iterations — do not wire new work through them unless explicitly asked to revive that approach.

Files/folders suffixed `- Copy` (e.g. `chat - Copy.py`, `vocab - Copy.pkl`) are manual backups the user made before an experiment; leave them alone unless asked to clean up.

## Architecture

**Data pipeline** (`message_cleaner.py` → `clean_tagged.txt` → `message_preprocess.py`):
1. `message_cleaner.py` converts a raw exported conversation into `clean_tagged.txt`, one line per merged turn, formatted as `<me> message text` / `<other> message text`.
2. `message_preprocess.py::load_dialog_stream()` reads that file and flattens the whole conversation into a **single continuous token stream**: `<speaker> word word ... <eot> <speaker> word ... <eot> ...`. This is not a pairs dataset — there's no fixed input/target split at load time.
3. `build_vocab_from_stream()` builds a frequency-capped vocab (default 19000 in `main.py`) with fixed special tokens `<pad>=0, <unk>=1, <eot>=2, <me>=3, <other>=4`.

**Training** (`main.py`):
- The token stream is turned into training examples via a sliding window (`StreamDataset` in `main.py`): for window size `MAX_LEN`, input is `data[i:i+MAX_LEN]` and target is `data[i+1:i+MAX_LEN+1]` — standard next-token-prediction over the whole conversation, not per-turn seq2seq.
- Train/val split is a random 90/10 split over windows (not a chronological split).
- Model: `DecoderOnlyTransformer` (`model.py`) — an `nn.TransformerEncoder` used autoregressively with a causal mask plus a padding mask, sinusoidal positional encoding, tied vocab-size output projection. This is a GPT-style architecture despite using `TransformerEncoderLayer` internally (there's no separate encoder/decoder).
- Loss is cross-entropy ignoring `<pad>`, with label smoothing; accuracy is computed over non-pad tokens only.
- Checkpoints save every epoch to `chatbot_model.pt` (overwritten each time, no versioning). Vocab is saved once at the end to `vocab.pkl`. Model hyperparameters (`d_model`, `nhead`, `num_layers`, etc.) are duplicated by hand in `main.py` and `chat.py` — if you change one, update the other or loading will fail via shape mismatch in `load_state_dict`.

**Inference** (`chat.py`):
- Conversation history is kept as a flat list of token ids (same `<speaker> ... <eot>` scheme as training), trimmed to `MAX_CONTEXT` tokens, snapping to the nearest `<eot>` boundary when trimming.
- Generation is autoregressive, one token at a time, with temperature scaling, top-k sampling, and a repetition penalty applied over the last 20 generated tokens. Generation stops early if the model emits `<eot>` (i.e., tries to end its turn).
- The user's turn is wrapped as `<me> ... <eot> <other>` before generation, so the model is explicitly cued whose turn it is next.

## Working in this repo

- `MAX_LEN`/`MAX_CONTEXT`, `d_model`, `nhead`, `num_layers`, `dim_feedforward`, and `pad_idx` must match between whatever produced `chatbot_model.pt` and whatever loads it (`main.py` vs `chat.py`). When changing model architecture, update both files together.
- The vocab is derived from the training corpus and vocab-size is a training hyperparameter (`VOCAB_LIMIT` in `main.py`) — regenerating `clean_tagged.txt` or changing `VOCAB_LIMIT` invalidates old checkpoints and `vocab.pkl`.
- `clean_tagged.txt`, `vocab.pkl`, `encoded_stream.pkl`, and any `data/` chat exports contain the user's real personal message history — treat as sensitive, don't print full contents or commit new exports carelessly.
