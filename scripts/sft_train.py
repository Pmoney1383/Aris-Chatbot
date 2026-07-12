"""SFT training loop for Aris — turns the Stage B base model into a chatbot.

Differences from pretraining (scripts/train.py, which stays untouched):
- starts from the latest Stage B checkpoint in checkpoints/ (weights only)
- trains on paired token/mask shards from data/sft_shards/
- masked loss: only assistant-turn tokens contribute to the gradient
- much lower LR (5e-5 -> 5e-6), short warmup (100), few steps (3000)
- no gradient checkpointing (SFT batches are small enough)
- checkpoints to checkpoints/sft/

--persona: tiny second pass on data/raw/clean_tagged_persona.txt (the user's
own iMessage logs), tokenized on the fly (no shards), LR 1e-5, 500 steps,
checkpoints to checkpoints/sft_persona/. Starts from the latest chat-SFT
checkpoint.

Same as train.py otherwise: bf16 autocast, 8-bit AdamW (bitsandbytes),
torch.compile with fallback, grad accumulation/clipping, cosine schedule,
rich live display, CSV loss log, auto-resume, Ctrl+C-safe checkpointing.
"""

import argparse
import csv
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
from src.config import GenerationConfig, ModelConfig, SFTConfig
from src.dataset import SFTShardedDataset, list_sft_shards
from src.model import DecoderOnlyTransformer

console = Console()

AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


# =========================================================
# PERSONA DATA (tokenized on the fly, no shards)
# =========================================================

def load_persona_stream(path: Path) -> tuple[list[int], list[int]]:
    """Parse clean_tagged.txt-style lines (<other> = user, <me> = assistant)
    into one chat-templated (ids, mask) stream. Consecutive lines from the
    same speaker are merged into a single turn."""
    turns: list[tuple[str, str]] = []
    role_map = {"<other>": "user", "<me>": "assistant"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            tag, _, text = line.partition(" ")
            role = role_map.get(tag)
            if role is None or not text:
                continue
            if turns and turns[-1][0] == role:
                turns[-1] = (role, turns[-1][1] + "\n" + text)
            else:
                turns.append((role, text))
    # stream must start with a user turn and end with an assistant turn
    while turns and turns[0][0] != "user":
        turns.pop(0)
    while turns and turns[-1][0] != "assistant":
        turns.pop()
    if not turns:
        raise ValueError(f"No usable <other>/<me> turns found in {path}")
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
# MASKED LOSS
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


# =========================================================
# LR SCHEDULE / CHECKPOINTING (same shape as train.py)
# =========================================================

def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


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


# =========================================================
# EVAL (masked)
# =========================================================

@torch.no_grad()
def evaluate(model, val_loader, device, n_batches, amp_dtype):
    model.eval()
    losses = []
    correct = 0
    total = 0
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
        preds = logits.argmax(dim=-1)
        correct += ((preds == y).float() * m).sum().item()
        total += m.sum().item()
    model.train()
    avg_loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


# =========================================================
# SAMPLING (chat format, stops at <|eot|>)
# =========================================================

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
    parser = argparse.ArgumentParser(description="SFT training for Aris")
    parser.add_argument(
        "--persona", action="store_true",
        help="fine-tune on data/raw/clean_tagged_persona.txt instead of SFT shards",
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    cfg = SFTConfig()
    gen_cfg = GenerationConfig()

    if args.persona:
        ckpt_dir = cfg.persona_checkpoint_dir
        # persona builds on the chat model, so prefer the SFT checkpoint
        init_dirs = [cfg.checkpoint_dir, cfg.pretrain_checkpoint_dir]
        max_lr, min_lr = cfg.persona_max_lr, cfg.persona_min_lr
        max_steps = cfg.persona_max_steps
        log_name = cfg.persona_log_name
    else:
        ckpt_dir = cfg.checkpoint_dir
        init_dirs = [cfg.pretrain_checkpoint_dir]
        max_lr, min_lr = cfg.max_lr, cfg.min_lr
        max_steps = cfg.max_steps
        log_name = cfg.log_name

    assert torch.cuda.is_available(), "CUDA GPU required for SFT training."
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    amp_dtype = AMP_DTYPES[cfg.amp_dtype]
    console.print(f"Mode: {'persona' if args.persona else 'chat SFT'} | "
                  f"AMP dtype: {cfg.amp_dtype} | lr {max_lr:.0e} -> {min_lr:.0e} | "
                  f"{max_steps} steps")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    if args.persona:
        ids, mask = load_persona_stream(cfg.persona_file)
        n_val_windows = max(1, len(ids) // model_cfg.max_seq_len // 20)  # ~5% val
        val_start = len(ids) - n_val_windows * model_cfg.max_seq_len - 1
        train_ds = PackedSFTDataset(ids[:val_start], mask[:val_start], model_cfg.max_seq_len)
        val_ds = PackedSFTDataset(ids[val_start:], mask[val_start:], model_cfg.max_seq_len)
        console.print(f"Persona stream: {len(ids):,} tokens "
                      f"({100 * sum(mask) / len(mask):.1f}% assistant), "
                      f"{len(train_ds)} train / {len(val_ds)} val windows")
    else:
        shards = list_sft_shards(cfg.shard_dir)
        if not shards:
            console.print(f"[red]No SFT shards in {cfg.shard_dir}. "
                          "Run scripts/prepare_sft_data.py first.[/]")
            sys.exit(1)
        if len(shards) >= 2:
            train_shards, val_shards = shards[:-1], shards[-1:]
        else:
            train_shards = val_shards = shards
        train_ds = SFTShardedDataset(train_shards, model_cfg.max_seq_len)
        val_ds = SFTShardedDataset(val_shards, model_cfg.max_seq_len)
        console.print(f"Shards: {len(train_shards)} train / {len(val_shards)} val "
                      f"({len(train_ds):,} train windows)")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # ---- model ----
    raw_model = DecoderOnlyTransformer(model_cfg).to(device)
    raw_model.get_num_params()

    # 8-bit Adam keeps optimizer state small (same as pretraining)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            raw_model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        console.print("Optimizer: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
            fused=cfg.fused_optimizer,
        )
        console.print("Optimizer: AdamW (bitsandbytes not available)")

    # ---- init weights / resume ----
    start_step = 0
    resume_path = latest_checkpoint(ckpt_dir)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        console.print(f"[bold yellow]Resuming SFT from {resume_path.name} "
                      f"at step {start_step} (loss {ckpt['loss']:.4f})[/]")
    else:
        init_path = next(
            (p for d in init_dirs if (p := latest_checkpoint(d)) is not None), None
        )
        if init_path is None:
            console.print(f"[red]No starting checkpoint found in "
                          f"{' or '.join(str(d) for d in init_dirs)} — "
                          "SFT needs pretrained weights.[/]")
            sys.exit(1)
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        console.print(f"[bold]Starting from base weights {init_path} "
                      f"(pretrain step {ckpt['step']}, loss {ckpt['loss']:.4f})[/]")

    # ---- compile ----
    model = raw_model
    if cfg.compile_model:
        try:
            model = torch.compile(raw_model)
            console.print("torch.compile: enabled")
        except Exception as e:
            model = raw_model
            console.print(f"[yellow]torch.compile: disabled ({e})[/]")
    else:
        console.print("torch.compile: disabled")

    # ---- loss log ----
    log_path = cfg.log_dir / log_name
    expected_log_header = [
        "step", "train_loss", "train_acc", "val_loss", "val_acc",
        "lr", "tokens_seen", "step_ms", "tokens_per_sec",
    ]
    new_log = not log_path.exists()
    if not new_log:
        with open(log_path, "r", newline="") as existing_log:
            old_header = next(csv.reader(existing_log), [])
        new_log = old_header != expected_log_header
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(expected_log_header)

    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * model_cfg.max_seq_len
    tokens_seen = start_step * tokens_per_step

    def format_eta(seconds):
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def status_table(step, loss, acc, val_loss, val_acc, lr, tps, step_time, eta_seconds):
        t = Table.grid(padding=(0, 2))
        acc_text = f"acc [bold]{acc:.4f}[/]" if acc is not None else "acc --"
        t.add_row(f"step [bold]{step}[/]/{max_steps}",
                  f"loss [bold]{loss:.4f}[/]",
                  acc_text,
                  f"val_loss [bold]{val_loss:.4f}[/]" if val_loss is not None else "val_loss --",
                  f"val_acc [bold]{val_acc:.4f}[/]" if val_acc is not None else "val_acc --",
                  f"lr {lr:.2e}",
                  f"tokens {tokens_seen / 1e6:.1f}M",
                  f"{tps / 1e3:.1f}k tok/s",
                  f"{step_time * 1000:.0f}ms/step",
                  f"ETA {format_eta(eta_seconds)}" if eta_seconds is not None else "ETA --")
        return t

    # ---- train ----
    model.train()
    step = start_step
    val_loss = None
    val_acc = None
    train_iter = iter(train_loader)
    last_loss = float("nan")
    avg_step_time = None

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while step < max_steps:
                t0 = time.time()
                lr = get_lr(step, max_lr, min_lr, cfg.warmup_steps, max_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0
                measure_train_acc = (
                    cfg.train_acc_every > 0
                    and (step + 1) % cfg.train_acc_every == 0
                )
                correct = 0
                total = 0

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

                    if measure_train_acc:
                        with torch.no_grad():
                            preds = logits.argmax(dim=-1)
                            correct += ((preds == y).float() * m).sum().item()
                            total += m.sum().item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                step += 1
                tokens_seen += tokens_per_step
                last_loss = loss_accum
                train_acc = correct / total if total else None
                torch.cuda.synchronize()
                step_time = time.time() - t0
                tps = tokens_per_step / step_time
                avg_step_time = (
                    step_time if avg_step_time is None
                    else 0.9 * avg_step_time + 0.1 * step_time
                )
                eta_seconds = avg_step_time * (max_steps - step)

                if step % cfg.eval_every == 0:
                    val_loss, val_acc = evaluate(
                        model, val_loader, device, cfg.val_batches, amp_dtype
                    )
                train_acc_log = f"{train_acc:.4f}" if train_acc is not None else ""
                if step % cfg.eval_every == 0:
                    log_writer.writerow([step, f"{loss_accum:.4f}", train_acc_log,
                                         f"{val_loss:.4f}", f"{val_acc:.4f}",
                                         f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                else:
                    log_writer.writerow([step, f"{loss_accum:.4f}", train_acc_log,
                                         "", "", f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                log_file.flush()

                if step % cfg.save_every == 0:
                    save_checkpoint(ckpt_dir, raw_model, optimizer,
                                    step, loss_accum, model_cfg)

                live.update(status_table(step, loss_accum, train_acc, val_loss, val_acc,
                                         lr, tps, step_time, eta_seconds))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving checkpoint...[/]")

    finally:
        path = save_checkpoint(ckpt_dir, raw_model, optimizer,
                               step, last_loss, model_cfg)
        console.print(f"Final checkpoint: {path}")
        log_file.close()

    # ---- sample conversations ----
    console.print("\n[bold]Sample responses:[/]\n")
    for prompt in ["Hi! How are you today?", "What can you help me with?"]:
        reply = sample_chat(raw_model, prompt, device, gen_cfg, amp_dtype)
        console.print(f"[cyan]user:[/] {prompt}")
        console.print(f"[white]assistant:[/] {reply}\n")
        console.print("-" * 60)


if __name__ == "__main__":
    main()
