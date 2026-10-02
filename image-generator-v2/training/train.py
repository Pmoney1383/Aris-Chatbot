"""
Trains the DiT on precomputed latents + caption embeddings with rectified flow.

Usage (from image-generator-v2/training):
    ../.venv/Scripts/python.exe train.py --phase 0    # overfit 64 samples (sanity test)
    ../.venv/Scripts/python.exe train.py --phase 1    # 256px
    ../.venv/Scripts/python.exe train.py --phase 2    # 512px, starts from phase 1's best checkpoint

Hyperparameters live in config.TRAIN_PHASES. Resumes automatically from the
newest checkpoint in checkpoints/dit/phase<N>/. Every save_every optimizer
steps it computes a deterministic validation loss with the EMA weights, keeps
best.pt when that improves, and saves a sample grid of the fixed validation
prompts to samples/phase<N>/. Ctrl+C saves a checkpoint before exiting.
Logs go to logs/dit_phase<N>.csv.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import signal
import sys
import time

import numpy as np
import torch
from PIL import Image

import config
import data
import flow
from dit import DiT, param_groups

sys.path.append(str(config.ROOT / "scripts"))
import console  # noqa: E402  (shared live-status-line helper)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def fmt_duration(seconds: float) -> str:
    m = int(seconds // 60)
    d, h, m = m // 1440, m // 60 % 24, m % 60
    return f"{d}d {h:02d}h{m:02d}m" if d else f"{h}h{m:02d}m"


def lr_lambda(cfg: config.TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def pad_texts(embs: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(e.shape[0] for e in embs)
    text = torch.zeros(len(embs), L, embs[0].shape[1], device=device)
    mask = torch.zeros(len(embs), L, dtype=torch.bool, device=device)
    for i, e in enumerate(embs):
        text[i, :e.shape[0]] = torch.from_numpy(e).float()
        mask[i, :e.shape[0]] = True
    return text, mask


def encode_prompts(prompts: list[str]) -> list[np.ndarray]:
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_ID)
    enc = T5EncoderModel.from_pretrained(config.TEXT_ENCODER_ID, torch_dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.inference_mode():
        for p in prompts:
            t = tok([p], max_length=config.PreprocessConfig().max_text_tokens, truncation=True, return_tensors="pt").to("cuda")
            out.append(enc(input_ids=t.input_ids).last_hidden_state[0].to(torch.float16).cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    return out


class EMA:
    """fp32 exponential moving average of the weights, kept in CPU RAM."""

    def __init__(self, model: torch.nn.Module):
        self.names = [n for n, _ in model.named_parameters()]
        self.shadow = [p.detach().float().cpu().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: float):
        for s, p in zip(self.shadow, model.parameters()):
            s.lerp_(p.detach().float().cpu(), 1 - decay)

    def state_dict(self):
        return dict(zip(self.names, self.shadow))

    def load_state_dict(self, sd):
        self.shadow = [sd[n].float().cpu().clone() for n in self.names]


def ema_model(ema: EMA, mcfg: config.ModelConfig) -> DiT:
    m = DiT(mcfg)
    m.load_state_dict(ema.state_dict(), strict=False)
    return m.cuda().to(torch.bfloat16).eval()


@torch.no_grad()
def validation_loss(model: DiT, val_ds, null: np.ndarray, cfg: config.TrainConfig) -> float:
    """Same samples, noise and timesteps every time, so values are comparable across checkpoints."""
    sampler = data.BucketBatchSampler(val_ds, 32, alpha=0.0, seed=12345)
    loader = torch.utils.data.DataLoader(val_ds, batch_sampler=sampler, num_workers=2,
                                         collate_fn=data.Collate(null, 0.0))
    gen = torch.Generator(device="cuda").manual_seed(12345)
    total, n = 0.0, 0
    for latents, text, mask, _ in loader:
        x0 = latents.cuda().float()
        t = (torch.arange(len(x0), device="cuda") + 0.5) / len(x0)
        noise = torch.randn(x0.shape, device="cuda", generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow.flow_loss(model, x0, text.cuda().float(), mask.cuda(), t=t, noise=noise)
        total += loss.item() * len(x0)
        n += len(x0)
    return total / max(n, 1)


@torch.no_grad()
def sample_grid(model: DiT, vae, prompt_embs, null: np.ndarray, size: tuple[int, int], cfg: config.TrainConfig, path):
    """size: (width, height) in pixels."""
    scfg = config.SampleConfig()
    text, mask = pad_texts(prompt_embs, "cuda")
    null_t, null_m = pad_texts([null], "cuda")
    gen = torch.Generator(device="cuda").manual_seed(config.VALIDATION_SEED)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = flow.sample(model, (len(prompt_embs), config.LATENT_CHANNELS, size[1] // config.VAE_DOWNSAMPLE, size[0] // config.VAE_DOWNSAMPLE), text, mask, null_t, null_m,
                        scfg.steps, cfg.sample_cfg_scale or scfg.cfg_scale, cfg.time_shift, gen)
    imgs = vae.decode(z.half() / vae.config.scaling_factor).sample
    imgs = imgs.float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    cols = 4
    rows = [np.concatenate(list(imgs[i:i + cols]), axis=1) for i in range(0, len(imgs), cols)]
    rows[-1] = np.pad(rows[-1], ((0, 0), (0, rows[0].shape[1] - rows[-1].shape[1]), (0, 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows)).save(path)


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=list(config.TRAIN_PHASES))
    args = p.parse_args()
    cfg = config.TRAIN_PHASES[args.phase]
    mcfg = config.ModelConfig()
    torch.manual_seed(cfg.seed)

    ckpt_dir = config.CHECKPOINT_DIR / f"phase{args.phase}"
    sample_dir = config.SAMPLE_DIR / f"phase{args.phase}"
    log_path = config.LOG_DIR / f"dit_phase{args.phase}.csv"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    overfit = cfg.max_samples is not None
    train_ds = data.LatentTextDataset(cfg.resolution, "train", cfg.val_per_mille, max_samples=cfg.max_samples,
                                      single_bucket=overfit, hflip=cfg.hflip, caption_variants=cfg.caption_variants)
    val_ds = None if overfit else data.LatentTextDataset(
        cfg.resolution, "val", cfg.val_per_mille, cfg.val_max_samples,
        caption_variants=(("detailed", 1.0),) if cfg.caption_variants else ())
    null = train_ds.text.null
    known_adult = data.adult_ids()
    prepared_adult = sum(sid in known_adult for _, _, sid in train_ds.items)
    print(f"phase {args.phase}: {len(train_ds):,} train samples at {cfg.resolution}px"
          + (f", {len(val_ds):,} validation" if val_ds else "")
          + f", {prepared_adult:,} verified-adult training samples", flush=True)

    model = DiT(mcfg).cuda()
    model.grad_checkpointing = cfg.grad_checkpointing
    import bitsandbytes as bnb  # imported here: importing it initialises CUDA, which DataLoader workers must not do
    opt = bnb.optim.AdamW8bit(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    ema = EMA(model)
    step = epoch = batch_in_epoch = 0
    best_val = float("inf")

    ckpt_path = latest_checkpoint(ckpt_dir)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, epoch, batch_in_epoch, best_val = ck["step"], ck["epoch"], ck["batch_in_epoch"], ck["best_val"]
        print(f"resumed from {ckpt_path.name} (step {step:,}, epoch {epoch})", flush=True)
    elif cfg.init_from_phase:
        candidates = [config.CHECKPOINT_DIR / f"phase{ph}" / "best.pt" for ph in cfg.init_from_phase]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            raise FileNotFoundError(f"phase {args.phase} starts from one of {[str(c) for c in candidates]}, but none exists")
        ck = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["ema"])
        ema.load_state_dict(ck["ema"])
        print(f"initialised from {src.parent.name} EMA weights (step {ck.get('step')})", flush=True)
    ck = None  # the loaded checkpoint (~4 GB of CPU tensors) is no longer needed
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiT {n_params / 1e6:.1f}M params | effective batch {cfg.micro_batch * cfg.grad_accum} "
          f"({cfg.micro_batch} x {cfg.grad_accum}) | {cfg.total_steps:,} optimizer steps", flush=True)

    if overfit:  # sample the training captions themselves: the model should reproduce those images
        prompt_embs = [train_ds.text.get(sid) for _, _, sid in train_ds.items[:8]]
        sample_size = tuple(int(v) for v in train_ds.bucket(0).split("x"))
    else:
        prompt_embs = encode_prompts(config.VALIDATION_PROMPTS)
        sample_size = (cfg.resolution, cfg.resolution)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.VAE_ID, torch_dtype=torch.float16).cuda().eval()

    fwd = torch.compile(model) if cfg.compile else model
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "epoch", "loss", "lr", "grad_norm", "samples_per_s", "val_loss", "elapsed_s"])

    def save(val_loss=None):
        nonlocal best_val
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "best_val": best_val,
              "model_config": dataclasses.asdict(mcfg), "train_config": dataclasses.asdict(cfg), "phase": args.phase}
        path = ckpt_dir / f"ckpt_{step:07d}.pt"
        torch.save(ck, path)
        for old in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-cfg.keep_last]:
            old.unlink()
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save({"ema": ema.state_dict(), "model_config": dataclasses.asdict(mcfg), "step": step,
                        "val_loss": val_loss, "resolution": cfg.resolution}, ckpt_dir / "best.pt")
        console.log(f"saved {path.name}" + (f" (new best val {val_loss:.4f} -> best.pt)"
                                            if val_loss is not None and val_loss <= best_val else ""))

    def evaluate():
        m = ema_model(ema, mcfg)
        val = validation_loss(m, val_ds, null, cfg) if val_ds else None
        sample_grid(m, vae, prompt_embs, null, sample_size, cfg, sample_dir / f"step_{step:07d}.png")
        del m
        torch.cuda.empty_cache()
        return val

    model.train()
    micro = 0
    win_loss, win_n, win_samples, win_start, start = torch.zeros((), device="cuda"), 0, 0, time.time(), time.time()
    step_loss = torch.zeros((), device="cuda")
    grad_norm = torch.tensor(0.0)
    loss_ema = sec_per_step = None
    last_step_time = time.time()
    last_val = best_val if best_val != float("inf") else None

    def status(n_batches: int) -> str:
        eta = fmt_duration((cfg.total_steps - step) * sec_per_step) if sec_per_step else "?"
        val = f" | val {last_val:.4f}" if last_val is not None else ""
        return (f"[phase {args.phase}] step {step:,}/{cfg.total_steps:,} ({100 * step / cfg.total_steps:.1f}%) | "
                f"epoch {epoch} batch {batch_in_epoch:,}/{n_batches:,} | loss {loss_ema:.4f}{val} | "
                f"lr {sched.get_last_lr()[0]:.2e} | ETA {eta}")

    try:
        while step < cfg.total_steps:
            sampler = data.BucketBatchSampler(train_ds, cfg.micro_batch, cfg.category_balance_alpha,
                                              cfg.seed, epoch, batch_in_epoch,
                                              adult_fraction=cfg.adult_sample_fraction)
            n_batches = len(sampler.batches(epoch))
            loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=True,
                collate_fn=data.Collate(null, cfg.caption_dropout), prefetch_factor=4 if cfg.num_workers else None)
            for latents, text, mask, _ in loader:
                x0 = latents.cuda(non_blocking=True).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = flow.flow_loss(fwd, x0, text.cuda(non_blocking=True).float(), mask.cuda(non_blocking=True),
                                          cfg.t_logit_mean, cfg.t_logit_std, cfg.time_shift)
                (loss / cfg.grad_accum).backward()
                micro += 1
                batch_in_epoch += 1
                win_loss += loss.detach()
                step_loss += loss.detach()
                win_n += 1
                win_samples += len(x0)
                if micro % cfg.grad_accum:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.ema_every == 0:
                    ema.update(model, min(cfg.ema_decay, (1 + step) / (10 + step)) ** cfg.ema_every)

                loss_val = step_loss.item() / cfg.grad_accum
                step_loss.zero_()
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
                now = time.time()
                dt, last_step_time = now - last_step_time, now
                sec_per_step = dt if sec_per_step is None else 0.95 * sec_per_step + 0.05 * dt
                console.live(status(n_batches))

                if step % cfg.log_every == 0:
                    avg = win_loss.item() / win_n
                    rate = win_samples / (time.time() - win_start)
                    mem = torch.cuda.max_memory_allocated() / 2**30
                    line = (f"step {step:,}/{cfg.total_steps:,} | epoch {epoch} | loss {avg:.4f} | "
                            f"lr {sched.get_last_lr()[0]:.2e} | grad {grad_norm.item():.3f} | {rate:,.1f} samples/s | "
                            f"peak {mem:.1f} GB")
                    if not console.IS_TTY:  # a terminal already shows the live line; logs get the detail
                        console.log(line)
                    log.writerow([step, epoch, f"{avg:.5f}", f"{sched.get_last_lr()[0]:.3e}",
                                  f"{grad_norm.item():.4f}", f"{rate:.2f}", "", f"{time.time() - start:.0f}"])
                    log_f.flush()
                    win_loss, win_n, win_samples, win_start = torch.zeros((), device="cuda"), 0, 0, time.time()

                if step % cfg.save_every == 0 or step >= cfg.total_steps:
                    console.log(f"step {step:,}: evaluating and saving...")
                    val = evaluate()
                    if val is not None:
                        last_val = val
                        console.log(f"step {step:,} | validation loss (EMA) {val:.4f}")
                        log.writerow([step, epoch, "", "", "", "", f"{val:.5f}", f"{time.time() - start:.0f}"])
                        log_f.flush()
                    save(val)
                    console.log(f"samples: {sample_dir / f'step_{step:07d}.png'}")
                    last_step_time = time.time()  # keep evaluation time out of the ETA
                if step >= cfg.total_steps:
                    break
            else:
                epoch += 1
                batch_in_epoch = 0
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not interrupt the save
        console.end_live()
        console.log("Ctrl+C: saving checkpoint (don't close the window)...")
        batch_in_epoch -= micro % cfg.grad_accum  # drop the unfinished accumulation; redo those batches on resume
        opt.zero_grad(set_to_none=True)
        save()
    finally:
        console.end_live()
        log_f.close()
    print(f"done at step {step:,}. Samples in {sample_dir}", flush=True)


if __name__ == "__main__":
    main()
