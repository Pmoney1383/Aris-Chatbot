"""Persona SFT v2 — fine-tune the chat-SFT model on the synthetic Aris
persona dataset (plus any manual curations from chat.py).

Data: data/raw/persona_synthetic.jsonl merged with data/raw/persona_curated.jsonl
(if it exists). One {"user": ..., "assistant": ...} pair per line, formatted
with the shared chat template (src/chat_format.py), loss masked to assistant
tokens only, packed into 1024-token windows, 90/10 train/val split by pair.

Every run starts fresh from checkpoints/sft/ckpt_003000.pt — never from a
previous persona checkpoint — and writes to checkpoints/sft_persona_v2/.
Early-stops if val loss rises 3 consecutive evals.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.chat_format import EOT_IDS, encode_chat_prompt, encode_conversation, get_encoding
from src.config import GenerationConfig, ModelConfig, PersonaV2Config
from src.model import DecoderOnlyTransformer

console = Console()

AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

SAMPLE_PROMPTS = [
    "who are you",
    "what is python",
    "yo what's up",
    "explain gravity",
]


# =========================================================
# DATA
# =========================================================

def load_pairs(cfg: PersonaV2Config) -> list[dict]:
    """Read persona_synthetic.jsonl (+ persona_curated.jsonl if present)."""
    sources = [cfg.synthetic_file]
    if cfg.curated_file.exists():
        sources.append(cfg.curated_file)

    pairs = []
    for path in sources:
        if not path.exists():
            console.print(f"[red]{path} not found. Run "
                          "scripts/generate_persona_data.py first.[/]")
            sys.exit(1)
        n_before = len(pairs)
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    console.print(f"[yellow]{path.name}:{line_no}: bad JSON, skipped[/]")
                    continue
                user = str(obj.get("user", "")).strip()
                assistant = str(obj.get("assistant", "")).strip()
                if user and assistant:
                    pairs.append({"user": user, "assistant": assistant})
        console.print(f"Loaded {len(pairs) - n_before:,} pairs from {path.name}")
    return pairs


def encode_pairs(pairs: list[dict]) -> tuple[list[int], list[int]]:
    """Chat-template every pair and concatenate into one (ids, mask) stream."""
    turns = []
    for p in pairs:
        turns.append(("user", p["user"]))
        turns.append(("assistant", p["assistant"]))
    return encode_conversation(turns)


class PackedSFTDataset(Dataset):
    """Non-overlapping (input, target, mask) windows over an in-memory stream."""

    def __init__(self, ids: list[int], mask: list[int], seq_len: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + self.seq_len + 1]
        m = self.mask[start + 1 : start + self.seq_len + 1]
        return x, y, m


# =========================================================
# LOSS / SCHEDULE / CHECKPOINTING (same shape as sft_train.py)
# =========================================================

def masked_loss(logits, targets, mask):
    """Cross-entropy over assistant tokens only (mask=1)."""
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    mask = mask.reshape(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(ckpt_dir, raw_model, optimizer, step, loss, model_cfg):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step:06d}.pt"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "model_config": model_cfg.__dict__,
    }, path)
    return path


@torch.no_grad()
def evaluate(model, val_loader, device, n_batches, amp_dtype):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(n_batches):
        try:
            x, y, m = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(x)
            loss = masked_loss(logits, y, m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_chat(raw_model, user_text, device, gen_cfg: GenerationConfig, amp_dtype):
    enc = get_encoding()
    x = torch.tensor([encode_chat_prompt(user_text)], dtype=torch.long, device=device)

    raw_model.eval()
    generated = []
    for _ in range(gen_cfg.max_new_tokens):
        x_cond = x[:, -raw_model.cfg.max_seq_len:]
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = raw_model(x_cond)
        logits = logits[:, -1, :].float()

        for tok in set(generated[-gen_cfg.repetition_window:]):
            logits[0, tok] /= gen_cfg.repetition_penalty

        logits /= gen_cfg.temperature
        v, _ = torch.topk(logits, gen_cfg.top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        generated.append(next_tok.item())
        x = torch.cat([x, next_tok], dim=1)
        if generated[-len(EOT_IDS):] == EOT_IDS:
            generated = generated[: -len(EOT_IDS)]
            break
    raw_model.train()
    return enc.decode(generated).strip()


# =========================================================
# MAIN
# =========================================================

def main():
    model_cfg = ModelConfig()
    cfg = PersonaV2Config()
    gen_cfg = GenerationConfig()

    assert torch.cuda.is_available(), "CUDA GPU required for persona training."
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    amp_dtype = AMP_DTYPES[cfg.amp_dtype]
    console.print(f"Persona SFT v2 | AMP dtype: {cfg.amp_dtype} | "
                  f"lr {cfg.max_lr:.0e} -> {cfg.min_lr:.0e} | {cfg.max_steps} steps")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    pairs = load_pairs(cfg)
    if not pairs:
        console.print("[red]No usable pairs found.[/]")
        sys.exit(1)

    import random
    random.Random(cfg.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_fraction))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    train_ids, train_mask = encode_pairs(train_pairs)
    val_ids, val_mask = encode_pairs(val_pairs)
    train_ds = PackedSFTDataset(train_ids, train_mask, model_cfg.max_seq_len)
    val_ds = PackedSFTDataset(val_ids, val_mask, model_cfg.max_seq_len)
    console.print(f"{len(train_pairs):,} train / {len(val_pairs):,} val pairs | "
                  f"{len(train_ids):,} / {len(val_ids):,} tokens | "
                  f"{len(train_ds):,} / {len(val_ds):,} windows")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # ---- model: always start fresh from the chat-SFT checkpoint ----
    if not cfg.init_checkpoint.exists():
        console.print(f"[red]Init checkpoint {cfg.init_checkpoint} not found. "
                      "Run scripts/sft_train.py first.[/]")
        sys.exit(1)

    raw_model = DecoderOnlyTransformer(model_cfg).to(device)
    raw_model.get_num_params()
    raw_model.gradient_checkpointing_enable()
    console.print("Gradient checkpointing: enabled")

    ckpt = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    console.print(f"[bold]Starting from {cfg.init_checkpoint} "
                  f"(SFT step {ckpt['step']}, loss {ckpt['loss']:.4f})[/]")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            raw_model.parameters(),
            lr=cfg.max_lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        console.print("Optimizer: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=cfg.max_lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        console.print("Optimizer: AdamW (bitsandbytes not available)")

    model = raw_model  # eager, same reasoning as sft_train (compile + ckpting OOMs)

    # ---- loss log (fresh file every run — this script never resumes) ----
    log_path = cfg.log_dir / cfg.log_name
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "lr",
                         "tokens_seen", "step_ms", "tokens_per_sec"])

    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * model_cfg.max_seq_len
    tokens_seen = 0

    def status_table(step, loss, val_loss, lr, tps, step_time, rises):
        t = Table.grid(padding=(0, 2))
        t.add_row(f"step [bold]{step}[/]/{cfg.max_steps}",
                  f"loss [bold]{loss:.4f}[/]",
                  f"val_loss [bold]{val_loss:.4f}[/]" if val_loss is not None else "val_loss --",
                  f"val rises {rises}/{cfg.early_stop_evals}",
                  f"lr {lr:.2e}",
                  f"tokens {tokens_seen / 1e6:.1f}M",
                  f"{tps / 1e3:.1f}k tok/s",
                  f"{step_time * 1000:.0f}ms/step")
        return t

    # ---- train ----
    model.train()
    step = 0
    val_loss = None
    prev_val_loss = None
    consecutive_rises = 0
    train_iter = iter(train_loader)
    last_loss = float("nan")

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while step < cfg.max_steps:
                t0 = time.time()
                lr = get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                for _ in range(cfg.grad_accum_steps):
                    try:
                        x, y, m = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y, m = next(train_iter)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, _ = model(x)
                        loss = masked_loss(logits, y, m)
                    loss = loss / cfg.grad_accum_steps
                    loss_accum += loss.item()
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                step += 1
                tokens_seen += tokens_per_step
                last_loss = loss_accum
                torch.cuda.synchronize()
                step_time = time.time() - t0
                tps = tokens_per_step / step_time

                if step % cfg.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device,
                                        cfg.val_batches, amp_dtype)
                    if prev_val_loss is not None and val_loss > prev_val_loss:
                        consecutive_rises += 1
                    else:
                        consecutive_rises = 0
                    prev_val_loss = val_loss
                    log_writer.writerow([step, f"{loss_accum:.4f}", f"{val_loss:.4f}",
                                         f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                else:
                    log_writer.writerow([step, f"{loss_accum:.4f}", "",
                                         f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                log_file.flush()

                if step % cfg.save_every == 0:
                    save_checkpoint(cfg.checkpoint_dir, raw_model, optimizer,
                                    step, loss_accum, model_cfg)

                live.update(status_table(step, loss_accum, val_loss, lr,
                                         tps, step_time, consecutive_rises))

                if consecutive_rises >= cfg.early_stop_evals:
                    console.print(f"\n[yellow]Early stop: val loss rose "
                                  f"{cfg.early_stop_evals} consecutive evals "
                                  f"(step {step}).[/]")
                    break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving checkpoint...[/]")

    finally:
        path = save_checkpoint(cfg.checkpoint_dir, raw_model, optimizer,
                               step, last_loss, model_cfg)
        console.print(f"Final checkpoint: {path}")
        log_file.close()

    # ---- sample responses ----
    console.print("\n[bold]Sample responses:[/]\n")
    for prompt in SAMPLE_PROMPTS:
        reply = sample_chat(raw_model, prompt, device, gen_cfg, amp_dtype)
        console.print(f"[cyan]user:[/] {prompt}")
        console.print(f"[white]assistant:[/] {reply}\n")
        console.print("-" * 60)


if __name__ == "__main__":
    main()
