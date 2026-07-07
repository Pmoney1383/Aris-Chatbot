"""Stage 0 training loop for Aris.

- configurable mixed precision, torch.compile (graceful fallback), AdamW
- cosine LR schedule with linear warmup
- gradient accumulation, gradient clipping
- checkpoint save/resume, CSV loss logging, rich live display
- prints sample continuations at the end
"""

import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.config import GenerationConfig, ModelConfig, TrainConfig
from src.dataset import ShardedDataset, split_shards
from src.model import DecoderOnlyTransformer

console = Console()


AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def get_amp_dtype(dtype_name: str):
    try:
        return AMP_DTYPES[dtype_name]
    except KeyError as exc:
        valid = ", ".join(sorted(AMP_DTYPES))
        raise ValueError(f"Unsupported amp_dtype {dtype_name!r}; expected one of: {valid}") from exc


# =========================================================
# LR SCHEDULE
# =========================================================

def get_lr(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


# =========================================================
# CHECKPOINTING
# =========================================================

def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(ckpt_dir, raw_model, optimizer, step, loss, model_cfg, scaler=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step:06d}.pt"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "loss": loss,
        "model_config": model_cfg.__dict__,
    }, path)
    return path


# =========================================================
# EVAL
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
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, loss = model(x, y)
        losses.append(loss.item())
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    model.train()
    avg_loss = sum(losses) / max(1, len(losses))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


# =========================================================
# SAMPLING
# =========================================================

@torch.no_grad()
def sample(raw_model, prompt, device, gen_cfg: GenerationConfig, amp_dtype, max_new_tokens=200):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    raw_model.eval()
    generated = []
    for _ in range(max_new_tokens):
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
    raw_model.train()
    return enc.decode(generated)


# =========================================================
# MAIN
# =========================================================

def main():
    model_cfg = ModelConfig()
    cfg = TrainConfig()
    gen_cfg = GenerationConfig()

    assert torch.cuda.is_available(), "CUDA GPU required for Stage 0 training."
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    amp_dtype = get_amp_dtype(cfg.amp_dtype)
    console.print(f"AMP dtype: {cfg.amp_dtype}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_shards, val_shards = split_shards(cfg.shard_dir, cfg.val_fraction)
    console.print(f"Shards: {len(train_shards)} train / {len(val_shards)} val")

    train_ds = ShardedDataset(train_shards, model_cfg.max_seq_len)
    val_ds = ShardedDataset(val_shards, model_cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ---- model ----
    raw_model = DecoderOnlyTransformer(model_cfg).to(device)
    raw_model.get_num_params()

    optimizer_kwargs = {
        "lr": cfg.max_lr,
        "betas": (0.9, 0.95),
        "weight_decay": cfg.weight_decay,
    }
    if cfg.fused_optimizer:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
        console.print("[yellow]fused AdamW unavailable; using standard AdamW[/]")
    else:
        console.print("Optimizer: AdamW fused" if cfg.fused_optimizer else "Optimizer: AdamW")

    scaler = torch.amp.GradScaler(device="cuda", enabled=(amp_dtype is torch.float16))

    # ---- resume ----
    start_step = 0
    ckpt_path = latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        console.print(f"[bold yellow]Resuming from {ckpt_path.name} "
                      f"at step {start_step} (loss {ckpt['loss']:.4f})[/]")
    else:
        console.print("[bold]No checkpoint found — starting from scratch.[/]")

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
    log_path = cfg.log_dir / "loss_log.csv"
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
        t.add_row(f"step [bold]{step}[/]/{cfg.max_steps}",
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
    avg_step_time = None  # exponential moving average, seconds/step

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while step < cfg.max_steps:
                t0 = time.time()
                lr = get_lr(step, cfg)
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
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, loss = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                    loss_accum += loss.item()
                    scaler.scale(loss).backward()

                    if measure_train_acc:
                        with torch.no_grad():
                            preds = logits.argmax(dim=-1)
                            correct += (preds == y).sum().item()
                            total += y.numel()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

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
                eta_seconds = avg_step_time * (cfg.max_steps - step)

                if step % cfg.eval_every == 0:
                    val_loss, val_acc = evaluate(
                        model, val_loader, device, cfg.val_batches, amp_dtype
                    )
                    train_acc_log = f"{train_acc:.4f}" if train_acc is not None else ""
                    log_writer.writerow([step, f"{loss_accum:.4f}", train_acc_log,
                                         f"{val_loss:.4f}", f"{val_acc:.4f}",
                                         f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                else:
                    train_acc_log = f"{train_acc:.4f}" if train_acc is not None else ""
                    log_writer.writerow([step, f"{loss_accum:.4f}", train_acc_log,
                                         "", "", f"{lr:.6e}", tokens_seen,
                                         f"{step_time * 1000:.1f}", f"{tps:.1f}"])
                log_file.flush()

                if step % cfg.save_every == 0:
                    save_checkpoint(cfg.checkpoint_dir, raw_model, optimizer,
                                    step, loss_accum, model_cfg, scaler)

                live.update(status_table(step, loss_accum, train_acc, val_loss, val_acc,
                                         lr, tps, step_time, eta_seconds))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving checkpoint...[/]")

    finally:
        path = save_checkpoint(cfg.checkpoint_dir, raw_model, optimizer,
                               step, last_loss, model_cfg, scaler)
        console.print(f"Final checkpoint: {path}")
        log_file.close()

    # ---- sample continuations ----
    console.print("\n[bold]Sample continuations:[/]\n")
    for prompt in ["Once upon a time", "The history of"]:
        text = sample(raw_model, prompt, device, gen_cfg, amp_dtype, max_new_tokens=200)
        console.print(f"[cyan]{prompt}[/][white]{text}[/]\n")
        console.print("-" * 60)


if __name__ == "__main__":
    main()
